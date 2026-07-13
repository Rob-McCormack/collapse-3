"""Ground-truth oracle: exact value labels and value-based regret.

The oracle wraps the exact solver as an *examiner* (not a player). For any
state it can label every legal move with the exact game-theoretic value of the
resulting position under optimal play, and from that derive **value-based
regret**: how far a move slides the outcome away from the best achievable one,
from the acting player's seat.

Why value-based (and not policy distance):
    Optimal play in Collapse3 is highly non-unique -- many moves preserve the
    same game value (e.g. every move that holds a draw). Measuring regret as the
    distance to one arbitrarily-chosen optimal policy, |pi_agent - pi_opt|,
    would flag correct-but-different moves as errors and inflate regret. Regret
    is therefore defined purely on outcome value: a move is optimal iff it
    achieves the state's value; its regret is the value it gives up.

Two regret scales are exposed:
    * ``regret``       -- raw solver-unit value drop (oriented to the mover):
                          0 optimal; 10/20/... attrition swings; 100+ if the
                          move flips a win/draw into a loss, etc.
    * ``class_regret`` -- integer steps down the ordinal outcome ladder
                          (true win > attrition win > draw > attrition loss >
                          true loss). Distribution-free and easy to bucket.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .game import (
    ATTRITION_LOSS,
    ATTRITION_WIN,
    DRAW,
    GameState,
    Move,
    TRUE_LOSS,
    TRUE_WIN,
    apply_move,
    get_legal_moves,
    orient,
)
from .solver import game_value

# Ordinal ladder from the acting player's seat (higher == better for mover).
_OUTCOME_RANK: Dict[int, int] = {
    TRUE_LOSS: 0,
    ATTRITION_LOSS: 1,
    DRAW: 2,
    ATTRITION_WIN: 3,
    TRUE_WIN: 4,
}

_OUTCOME_NAME: Dict[int, str] = {
    TRUE_LOSS: "true loss",
    ATTRITION_LOSS: "attrition loss",
    DRAW: "draw",
    ATTRITION_WIN: "attrition win",
    TRUE_WIN: "true win",
}


def outcome_name(oriented_value: int) -> str:
    return _OUTCOME_NAME.get(oriented_value, f"value {oriented_value}")


@dataclass(frozen=True)
class MoveLabel:
    move: Move
    child_value: int      # exact P0-perspective value of the resulting state
    mover_value: int      # child_value oriented to the player who moved
    regret: int           # raw value drop from the mover's seat (>= 0)
    class_regret: int     # steps down the ordinal outcome ladder (>= 0)
    is_optimal: bool


@dataclass(frozen=True)
class StateLabel:
    state: GameState
    mover: int
    state_value: int          # P0-perspective value of the state
    best_mover_value: int     # state value oriented to the mover
    moves: Tuple[MoveLabel, ...]

    @property
    def optimal_moves(self) -> Tuple[Move, ...]:
        return tuple(m.move for m in self.moves if m.is_optimal)

    @property
    def is_critical(self) -> bool:
        """A decision is *critical* if at least one legal move is a mistake --
        i.e. not all moves preserve the state's value. On non-critical states
        every move is optimal and the agent cannot lower its outcome."""
        return any(not m.is_optimal for m in self.moves)


class Oracle:
    """Exact examiner over Collapse3 states.

    Reuses the solver's transposition table across calls, so labelling many
    states from the same game family is cheap after the first solve.
    """

    def state_value(self, state: GameState) -> int:
        return game_value(state)

    def label_state(self, state: GameState) -> Optional[StateLabel]:
        """Label every legal move of ``state``.

        Returns ``None`` for terminal states or states with no legal move
        (there is nothing to decide).
        """
        moves = get_legal_moves(state)
        if not moves:
            return None

        mover = state.turn
        state_val = game_value(state)
        best_mover_val = orient(state_val, mover)

        labels: List[MoveLabel] = []
        for move in moves:
            child_val = game_value(apply_move(state, move))
            mover_val = orient(child_val, mover)
            regret = best_mover_val - mover_val
            class_regret = _OUTCOME_RANK[best_mover_val] - _OUTCOME_RANK[mover_val]
            labels.append(
                MoveLabel(
                    move=move,
                    child_value=child_val,
                    mover_value=mover_val,
                    regret=regret,
                    class_regret=class_regret,
                    is_optimal=(regret == 0),
                )
            )

        return StateLabel(
            state=state,
            mover=mover,
            state_value=state_val,
            best_mover_value=best_mover_val,
            moves=tuple(labels),
        )

    def move_regret(self, state: GameState, move: Move) -> int:
        """Raw value-based regret of a single move (mover's seat, >= 0)."""
        best = orient(game_value(state), state.turn)
        got = orient(game_value(apply_move(state, move)), state.turn)
        return best - got

    def is_optimal_move(self, state: GameState, move: Move) -> bool:
        return self.move_regret(state, move) == 0
