"""Full-size exact grading (experiments/horizon.py) and the root solve.

Locks four things:

1. THE ROOT: capped_solve determines the full 14-bead game's opening value --
   +100, a first-player forced alignment win -- within a small node budget.
   (Cross-checks that motivated trusting this: the main solver agrees from a
   fresh table; an independent alpha-beta with NO transposition table and NO
   symmetry folding, sharing only the rules engine, returned +100 in 42.2M
   nodes; the principal variation was re-verified position-by-position with
   fresh solves; the diagonal (6,6)..(14,14) and off-diagonal mover-win
   pattern are continuous with the published small-size table.)

2. THE GATE: kimi-v2 self-play at (14,14) is deterministic. The trajectory,
   exact tail values, and grading reproduce the external session that first
   probed this game: 14 plies, final -100, and plies 6, 7, 8 each a thrown
   forced win (WDL regret 2). The session's solver could not solve plies 0-3
   within 4M nodes; the repo's solver does the ROOT in ~96K. The values agree
   at every ply both solved, so the difference is search strength (symmetry
   folding + ordering), not engine semantics.

3. THE CONSISTENCY INVARIANT: along a trajectory the exact value may change
   across a move only if that move is graded with positive regret (the
   mover-oriented value can never improve across the mover's own move).
   analyze_game asserts this over every solved tail; these tests exercise it
   on real trajectories and on a synthetic violation.

4. Recorded-JSON guards on the shipped battery distributions.
"""

import json
import os
from pathlib import Path

import pytest

from collapse3.game import empty_state
from collapse3.solver import NodeCapExceeded, capped_solve, game_value
from experiments.horizon import PAIRINGS, analyze_game, play_trajectory

RESULTS = Path(__file__).resolve().parent.parent / "results" / "horizon_latest.json"
SLOW = os.environ.get("COLLAPSE3_SLOW") == "1"

START_14 = empty_state(14, 14)

# Session gate values (kimi-v2 self-play, plies 4..13) -- the tail the weaker
# session solver could reach. The repo solver extends them back to ply 0.
SESSION_TAIL = {4: 100, 5: 100, 6: 100, 7: -100, 8: 100,
                9: -100, 10: -100, 11: -100, 12: -100, 13: -100}


def test_root_of_full_game_is_first_player_win():
    value, nodes = capped_solve(START_14, 200_000)
    assert value == 100
    assert nodes < 200_000


def test_capped_solve_agrees_with_main_solver_at_small_sizes():
    for r0, r1 in [(3, 3), (4, 3), (4, 4), (5, 5)]:
        s = empty_state(r0, r1)
        assert capped_solve(s, 10_000_000)[0] == game_value(s, fresh=True)


def test_capped_solve_raises_on_cap():
    with pytest.raises(NodeCapExceeded):
        capped_solve(START_14, 1_000)


def test_kimi_v2_selfplay_gate_and_invariant():
    """The Part-4 reproduction gate, extended: horizon is ply 0, not ply 4."""
    p0, p1 = PAIRINGS["kimi-v2/kimi-v2"][0](0)
    states, moves = play_trajectory(p0, p1, START_14)
    assert len(moves) == 14

    # analyze_game asserts the consistency invariant internally at every ply.
    rec = analyze_game(states, moves, 500_000)
    assert rec["final_value"] == -100
    assert rec["horizon_ply"] == 0            # the whole game is graded
    assert rec["graded_fraction"] == 1.0

    for ply, v in SESSION_TAIL.items():
        assert rec["tail_values"][str(ply)] == v, f"value drift at ply {ply}"

    regrets = {g["ply"]: g["wdl_regret"] for g in rec["grading"]}
    assert {p for p, r in regrets.items() if r > 0} == {6, 7, 8}
    assert all(regrets[p] == 2 for p in (6, 7, 8))  # three thrown forced wins
    assert rec["blunders"] == 3


def test_invariant_rejects_unflagged_cutoff_style_corruption():
    """A value flip across a zero-regret move must raise -- the exact failure
    mode of a transposition table that stores cutoffs without bound flags."""
    p0, p1 = PAIRINGS["kimi-v2/kimi-v2"][0](0)
    states, moves = play_trajectory(p0, p1, START_14)

    import experiments.horizon as hz
    real = hz.capped_solve
    # Corrupt ply 13 (P1 to move; true value -100, P1 winning) to read +100.
    # Across the ply-13 move the value then "improves" for the mover
    # (-100 -> won game becomes +100 -> lost-for-P1 reading), which is exactly
    # the impossible trajectory an unflagged-cutoff table produces.
    target = states[13]

    def corrupted(state, cap):
        if state == target:
            return 100, 1
        return real(state, cap)

    hz.capped_solve = corrupted
    try:
        with pytest.raises(AssertionError):
            analyze_game(states, moves, 500_000)
    finally:
        hz.capped_solve = real


def test_recorded_battery_distributions():
    rec = json.loads(RESULTS.read_text())
    pairings = rec["results"]["pairings"]
    assert rec["provenance"]["config"]["node_cap"] == 4_000_000

    for name, r in pairings.items():
        # Every game in every pairing was solved back to the opening.
        assert all(h == 0 for h in r["horizons"]), name
        assert all(f == 1.0 for f in r["graded_fractions"]), name

    # The kimi self-play games all end in 14 plies with 3 thrown forced wins.
    for name in ("kimi-v2/kimi-v2", "kimi-v1/kimi-v1"):
        assert pairings[name]["lengths"] == [14]
        assert pairings[name]["blunder_counts"] == [3]


@pytest.mark.skipif(not SLOW, reason="set COLLAPSE3_SLOW=1")
def test_diagonal_first_player_win_6_through_14():
    for r in range(6, 15):
        v, _ = capped_solve(empty_state(r, r), 30_000_000)
        assert v == 100, f"({r},{r}) expected +100, got {v}"
