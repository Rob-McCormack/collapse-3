"""Guards on the full-game root solve (experiments/full_game_value.py).

The (14,14) root is a first-player forced TRUE win (+100). Verified four
independent ways before these guards were written:

  * main solver (bound-flagged TT + D4 symmetry folding): +100, ~96K nodes;
  * raw alpha-beta, NO transposition table, NO symmetry folding: +100,
    42,236,657 nodes (recorded in the results JSON, not rerun by default);
  * clean-room reference rules engine + separate plain-TT alpha-beta
    (zero shared code with the main engine): +100, ~126K nodes;
  * principal variation re-verified position-by-position with fresh solves.

The 14x14 opening grid extends the published 6x6 table without changing a
single previously published cell (cross-checked at write time, and guarded
against the shipped opening_values JSON below).
"""

import json
from pathlib import Path

from collapse3.game import empty_state
from collapse3.solver import capped_solve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FULL = RESULTS_DIR / "full_game_value_latest.json"
OPENING = RESULTS_DIR / "opening_values_latest.json"


def _grid():
    return json.loads(FULL.read_text())["results"]


def test_live_root_value_is_first_player_true_win():
    value, nodes = capped_solve(empty_state(14, 14), 200_000)
    assert value == 100
    rec = _grid()
    assert rec["root_14_14"]["value"] == 100
    assert rec["reference_engine_14_14"]["value"] == 100
    assert rec["no_tt_no_symmetry_probe"]["value"] == 100


def test_recorded_grid_structure():
    grid = _grid()["grid"]
    # Diagonal: draws through (5,5), first-player true win from (6,6) up.
    for r in range(1, 6):
        assert grid[str(r)][str(r)] == 0
    for r in range(6, 15):
        assert grid[str(r)][str(r)] == 100
    # From r0 >= 6: attrition win vs 1-2 opposing reserves, true win vs 3+.
    for r0 in range(6, 15):
        for r1 in range(1, 15):
            expected = 10 if r1 <= 2 else 100
            assert grid[str(r0)][str(r1)] == expected, (r0, r1)
    # Below-diagonal small rows: first-player attrition win; above: draw.
    for r0 in range(2, 6):
        for r1 in range(1, 15):
            expected = 10 if r1 < r0 else 0
            assert grid[str(r0)][str(r1)] == expected, (r0, r1)
    # Row 1 is all draws.
    assert all(v == 0 for v in grid["1"].values())


def test_grid_agrees_with_published_opening_values():
    grid = _grid()["grid"]
    old = json.loads(OPENING.read_text())["results"]["grid"]
    assert len(old) == 36
    for key, cell in old.items():
        r0, r1 = key.strip("()").split(",")
        assert grid[r0][r1] == cell["value"], key


def test_recorded_pv_was_verified():
    rec = _grid()
    assert rec["pv_14_14"]["verified"] is True
    assert rec["pv_14_14"]["plies"] == len(rec["pv_14_14"]["moves"])
