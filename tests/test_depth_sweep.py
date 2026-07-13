"""Competence-per-ply: locks in the 'no shallow basic strategy' finding.

At (4,4) from the drawn P0 seat, against an optimal opponent (Finding 8):
  * the naive own-alignment heuristic is optimal on *zero* critical decisions;
  * a one-ply bead-count search carries no signal -- it equals random play,
    because every legal move shifts the material differential by the same +1
    (a placement adds one of yours, a removal deletes one of theirs);
  * a *better* one-ply evaluation helps but still loses the drawn game;
  * one extra ply of lookahead over that same evaluation is decisive.
"""

import json
from pathlib import Path

from collapse3.agents import (
    HeuristicNPlyAgent,
    MyopicAgent,
    NPlyAgent,
    OptimalAgent,
    RandomAgent,
)
from collapse3.game import apply_move, empty_state, get_legal_moves
from collapse3.metrics import evaluate_competence

_N = 40
_SEED = 12345
_RESULTS = Path(__file__).resolve().parent.parent / "results" / "depth_sweep_latest.json"


def _beads(board):
    return sum(c.count(0) for c in board), sum(c.count(1) for c in board)


def test_every_move_shifts_bead_differential_by_plus_one():
    # The invariant that makes a one-ply bead-count eval blind: from the mover's
    # seat, every legal move raises (my beads - their beads) by exactly +1.
    s = empty_state(4, 4, 0)
    seen_placement = seen_removal = False
    for _ in range(6):  # walk a few plies deep enough to expose removals
        mover = s.turn
        p_before = _beads(s.board)
        diff_before = p_before[mover] - p_before[1 - mover]
        for m in get_legal_moves(s):
            b = _beads(apply_move(s, m).board)
            assert (b[mover] - b[1 - mover]) - diff_before == 1
            if b[mover] > p_before[mover]:
                seen_placement = True
            if b[1 - mover] < p_before[1 - mover]:
                seen_removal = True
        s = apply_move(s, get_legal_moves(s)[0])
    assert seen_placement and seen_removal  # both move types were exercised


def _crit(test_factory, n_games=_N, seed=_SEED):
    r = evaluate_competence(
        test_factory=test_factory,
        opp_factory=lambda s: OptimalAgent(seed=s),
        res0=4, res1=4, n_games=n_games, base_seed=seed, test_seat=0,
    )
    return r.optimal_rate_critical, r.loss_rate


def test_myopic_greed_is_below_random_on_critical_decisions():
    crit, loss = _crit(lambda s: MyopicAgent(seed=s))
    assert crit == 0.0
    assert loss > 0.9


def test_one_ply_bead_count_search_equals_random():
    rand_crit, _ = _crit(lambda s: RandomAgent(seed=s))
    nply_crit, _ = _crit(lambda s: NPlyAgent(depth=1, seed=s))
    assert nply_crit == rand_crit


def test_better_eval_at_one_ply_helps_but_still_loses_the_draw():
    rand_crit, _ = _crit(lambda s: RandomAgent(seed=s))
    pos_crit, pos_loss = _crit(lambda s: HeuristicNPlyAgent(depth=1, seed=s))
    assert pos_crit > rand_crit    # a working eval has signal
    assert pos_loss > 0.4          # ...but no one-ply snapshot heuristic holds the draw


def test_recorded_positional_one_extra_ply_headline():
    """Lock the README's 0.62 -> 0.997 critical-accuracy jump to shipped results.

    The headline trajectory is the *positional* (threat-aware) eval at depth
    1 -> 2 vs the optimal opponent, from the 200-game run in the results JSON.
    """
    rows = {r["policy"]: r
            for r in json.loads(_RESULTS.read_text())["results"]["rows"]
            if r["opponent"] == "optimal"}
    assert abs(rows["pos-1ply"]["optimal_rate_critical"] - 0.62) < 0.01
    assert abs(rows["pos-2ply"]["optimal_rate_critical"] - 0.997) < 0.005
    assert abs(rows["pos-1ply"]["loss_rate"] - 0.64) < 0.01


def test_one_extra_ply_of_lookahead_is_decisive():
    pos1, loss1 = _crit(lambda s: HeuristicNPlyAgent(depth=1, seed=s))
    pos2, loss2 = _crit(lambda s: HeuristicNPlyAgent(depth=2, seed=s))
    assert pos2 > 0.95            # near-perfect on critical decisions
    assert pos2 > pos1 + 0.3      # a single ply is a large, decisive jump
    assert loss2 < loss1