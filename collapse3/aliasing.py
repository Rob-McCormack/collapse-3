"""Representation / aliasing regret floors.

Given the exact value map, this measures the *irreducible* regret a memoryless
agent must pay under a lossy observation -- the best any policy could do if it
can only see the observation, not the full state.

Method (exhaustive, exact): enumerate all decision states, group them by their
lossy observation, and for each group pick the single action that is legal in
*every* member and minimises total win/draw/loss regret. Groups with no action
legal in all members contribute 0 (a charity rule that keeps the result a true
lower bound). The mean over all decisions is the floor.

Regret is in win/draw/loss units (win=1, draw=0, loss=-1), so the maximum regret
per decision is 2, matching the representation-study convention.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .enumeration import wdl
from .game import Board, GameState, Move, apply_move, evaluate_terminal, get_legal_moves

ObsFn = Callable[[GameState], Tuple]


def beads_on_board(board: Board, player: int) -> int:
    return sum(col.count(player) for col in board)


def memory_reserves_obs(initial_res: Tuple[int, int]) -> ObsFn:
    """Reserve-blind snapshot plus destroyed counts derived from full state."""

    def obs(state: GameState) -> Tuple:
        on0 = beads_on_board(state.board, 0)
        on1 = beads_on_board(state.board, 1)
        destroyed = (
            initial_res[0] - state.res[0] - on0,
            initial_res[1] - state.res[1] - on1,
        )
        return (state.board, state.turn, state.cooldown, destroyed)

    return obs

# Standard lossy observations (history-created fields removed from a snapshot).
OBSERVATIONS: Dict[str, ObsFn] = {
    "full": lambda s: (s.board, s.res, s.turn, s.cooldown),
    "hide_cooldown": lambda s: (s.board, s.res, s.turn),
    "hide_reserves": lambda s: (s.board, s.turn, s.cooldown),
    "hide_both": lambda s: (s.board, s.turn),
}


@dataclass
class FloorResult:
    floor: float
    n_decisions: int
    obs_groups: int
    aliased_groups: int
    conflict_groups: int
    no_common_legal_action: int
    max_states_per_obs: int
    per_state_regret: Dict[GameState, float] = field(default_factory=dict)


def regret_floor(memo: Dict[GameState, int], obs_fn: ObsFn, p0_only: bool = False) -> FloorResult:
    """Irreducible regret floor of the best memoryless policy on ``obs_fn``.

    ``p0_only`` restricts the census to P0 decisions (the agent-relevant subset
    when P0 is the learner).
    """
    groups: Dict[Tuple, List[Tuple[GameState, Dict[Move, int], int]]] = {}
    n_decisions = 0
    for s in memo:
        if evaluate_terminal(s) is not None:
            continue
        if p0_only and s.turn != 0:
            continue
        moves = get_legal_moves(s)
        if not moves:
            continue
        n_decisions += 1
        mover = s.turn
        avals = {m: wdl(memo[apply_move(s, m)], mover) for m in moves}
        vstar = max(avals.values())
        groups.setdefault(obs_fn(s), []).append((s, avals, vstar))

    per_state: Dict[GameState, float] = {}
    floor_total = 0.0
    aliased = conflict = no_common = 0
    max_group = 0

    for members in groups.values():
        max_group = max(max_group, len(members))
        if len(members) > 1:
            aliased += 1
            inter = None
            for _, avals, vstar in members:
                opt = {m for m, u in avals.items() if u == vstar}
                inter = opt if inter is None else (inter & opt)
            if not inter:
                conflict += 1

        candidates = set()
        for _, avals, _ in members:
            candidates |= set(avals)

        best_cand, best_tot = None, None
        for cand in candidates:
            tot, ok = 0.0, True
            for _, avals, vstar in members:
                if cand not in avals:
                    ok = False
                    break
                tot += vstar - avals[cand]
            if ok and (best_tot is None or tot < best_tot):
                best_cand, best_tot = cand, tot

        if best_cand is None:
            no_common += 1  # charity: contribute 0, preserving a lower bound

        for s, avals, vstar in members:
            r = (vstar - avals[best_cand]) if best_cand is not None else 0.0
            per_state[s] = r
            floor_total += r

    return FloorResult(
        floor=floor_total / n_decisions if n_decisions else 0.0,
        n_decisions=n_decisions,
        obs_groups=len(groups),
        aliased_groups=aliased,
        conflict_groups=conflict,
        no_common_legal_action=no_common,
        max_states_per_obs=max_group,
        per_state_regret=per_state,
    )
