"""Competence measurement for Collapse3.

Separates the two quantities the write-up is about:

    performance  -- win rate against a specific opponent (distribution-dependent).
    competence   -- value-based regret against the oracle (distribution-free per
                    decision, but sampled over whatever state distribution the
                    opponent induces).

Key reported quantities:
    * outcome tally / win-rate (from the test agent's seat),
    * optimal-move rate, overall and restricted to *critical* decisions
      (states where at least one legal move is a mistake),
    * mean value-based regret, overall and on critical decisions,
    * a histogram over the ordinal outcome-ladder regret (class_regret).

The overall-vs-critical split is the point: aggregate optimal-move rate is
adversarially flattering because most states have many optimal moves.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .agents import Agent
from .enumeration import wdl
from .game import (
    GameState,
    Move,
    attrition_value,
    apply_move,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    orient,
)
from .oracle import Oracle, outcome_name
from .solver import game_value

AgentFactory = Callable[[int], Agent]


@dataclass
class Decision:
    ply: int
    mover: int
    regret: int          # weighted solver-score units (draw->true-loss = 100)
    wdl_regret: int       # win/draw/loss units (0/1/2); comparable to aliasing floors
    class_regret: int
    is_optimal: bool
    is_critical: bool


@dataclass
class GameResult:
    terminal_value: int          # P0 perspective
    plies: int
    decisions: List[Decision] = field(default_factory=list)


def play_game(
    agent0: Agent,
    agent1: Agent,
    start: GameState,
    oracle: Optional[Oracle] = None,
    track_seats: Sequence[int] = (0, 1),
    max_plies: int = 400,
) -> GameResult:
    """Play one game to completion, optionally recording per-decision regret.

    Regret is recorded only for the seats in ``track_seats`` and only when an
    ``oracle`` is supplied (labelling is the expensive part).
    """
    agents = (agent0, agent1)
    state = start
    decisions: List[Decision] = []

    for ply in range(max_plies):
        terminal = evaluate_terminal(state)
        if terminal is not None:
            return GameResult(terminal, ply, decisions)
        moves = get_legal_moves(state)
        if not moves:
            return GameResult(attrition_value(state.board), ply, decisions)

        mover = state.turn
        move = agents[mover].choose(state)

        if oracle is not None and mover in track_seats:
            label = oracle.label_state(state)
            ml = next(m for m in label.moves if m.move == move)
            wdl_regret = wdl(label.state_value, mover) - wdl(ml.child_value, mover)
            decisions.append(
                Decision(
                    ply=ply,
                    mover=mover,
                    regret=ml.regret,
                    wdl_regret=wdl_regret,
                    class_regret=ml.class_regret,
                    is_optimal=ml.is_optimal,
                    is_critical=label.is_critical,
                )
            )
        state = apply_move(state, move)

    # Safety net; games are provably finite, so this should not trigger.
    final = evaluate_terminal(state)
    return GameResult(final if final is not None else attrition_value(state.board), max_plies, decisions)


@dataclass
class CompetenceReport:
    games: int
    res: Tuple[int, int]
    test_seat: int
    outcomes: Dict[str, int]          # from the test agent's perspective
    decisions: int
    optimal_decisions: int
    critical_decisions: int
    optimal_on_critical: int
    total_regret: int
    total_regret_critical: int
    total_wdl_regret: int
    total_wdl_regret_critical: int
    class_regret_hist: Dict[int, int]

    @property
    def win_rate(self) -> float:
        wins = self.outcomes.get("true win", 0) + self.outcomes.get("attrition win", 0)
        return wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.outcomes.get("draw", 0) / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        losses = self.outcomes.get("true loss", 0) + self.outcomes.get("attrition loss", 0)
        return losses / self.games if self.games else 0.0

    @property
    def optimal_rate(self) -> float:
        return self.optimal_decisions / self.decisions if self.decisions else float("nan")

    @property
    def optimal_rate_critical(self) -> float:
        return self.optimal_on_critical / self.critical_decisions if self.critical_decisions else float("nan")

    @property
    def mean_regret(self) -> float:
        return self.total_regret / self.decisions if self.decisions else float("nan")

    @property
    def mean_regret_critical(self) -> float:
        return self.total_regret_critical / self.critical_decisions if self.critical_decisions else float("nan")

    @property
    def mean_wdl_regret(self) -> float:
        return self.total_wdl_regret / self.decisions if self.decisions else float("nan")

    @property
    def mean_wdl_regret_critical(self) -> float:
        return self.total_wdl_regret_critical / self.critical_decisions if self.critical_decisions else float("nan")

    def summary(self) -> str:
        lines = [
            f"games={self.games}  reserves={self.res}  test_seat=P{self.test_seat}",
            f"  outcomes (test seat): {dict(sorted(self.outcomes.items()))}",
            f"  win/draw/loss: {self.win_rate:.3f} / {self.draw_rate:.3f} / {self.loss_rate:.3f}",
            f"  decisions: {self.decisions}  (critical: {self.critical_decisions})",
            f"  optimal-move rate:  overall {self.optimal_rate:.4f}   critical-only {self.optimal_rate_critical:.4f}",
            f"  mean regret:        overall {self.mean_regret:.4f}   critical-only {self.mean_regret_critical:.4f}",
            f"  mean regret (WDL):  overall {self.mean_wdl_regret:.4f}   critical-only {self.mean_wdl_regret_critical:.4f}",
            f"  class-regret hist:  {dict(sorted(self.class_regret_hist.items()))}",
        ]
        return "\n".join(lines)


def evaluate_competence(
    test_factory: AgentFactory,
    opp_factory: AgentFactory,
    res0: int,
    res1: int,
    n_games: int,
    base_seed: int = 0,
    test_seat: int = 0,
    oracle: Optional[Oracle] = None,
) -> CompetenceReport:
    """Measure a test agent's competence against an opponent distribution.

    The test agent occupies ``test_seat``; the opponent occupies the other seat.
    Fresh, per-game-seeded agents are built from the factories so the run is
    fully reproducible from ``(base_seed, n_games)``.
    """
    oracle = oracle or Oracle()
    start = empty_state(res0, res1)
    game_value(start)  # warm the shared transposition table once

    outcomes: Counter = Counter()
    decisions = optimal = critical = optimal_on_critical = 0
    total_regret = total_regret_critical = 0
    total_wdl = total_wdl_critical = 0
    class_hist: Counter = Counter()

    for g in range(n_games):
        test_agent = test_factory(base_seed + g)
        opp_agent = opp_factory(base_seed + 10_000 + g)
        if test_seat == 0:
            a0, a1 = test_agent, opp_agent
        else:
            a0, a1 = opp_agent, test_agent

        result = play_game(a0, a1, start, oracle=oracle, track_seats=(test_seat,))
        outcomes[outcome_name(orient(result.terminal_value, test_seat))] += 1

        for d in result.decisions:
            decisions += 1
            total_regret += d.regret
            total_wdl += d.wdl_regret
            class_hist[d.class_regret] += 1
            if d.is_optimal:
                optimal += 1
            if d.is_critical:
                critical += 1
                total_regret_critical += d.regret
                total_wdl_critical += d.wdl_regret
                if d.is_optimal:
                    optimal_on_critical += 1

    return CompetenceReport(
        games=n_games,
        res=(res0, res1),
        test_seat=test_seat,
        outcomes=dict(outcomes),
        decisions=decisions,
        optimal_decisions=optimal,
        critical_decisions=critical,
        optimal_on_critical=optimal_on_critical,
        total_regret=total_regret,
        total_regret_critical=total_regret_critical,
        total_wdl_regret=total_wdl,
        total_wdl_regret_critical=total_wdl_critical,
        class_regret_hist=dict(class_hist),
    )
