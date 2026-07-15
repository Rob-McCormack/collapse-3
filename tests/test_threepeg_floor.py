"""Three-Peg Collapse sibling variant: reproduce gate, exactness lemma, geometry.

Guards Fable's Part A claims. Everything is deterministic and enumerable at every
reserve size, so the whole 2..14 sweep runs in the default suite (~30-60s):

  * the A2 reproduce gate (state/decision counts + both single-hide floors) at
    every size -- any mismatch is engine drift and voids the finding;
  * the opening gate, including the certified centre-strategy inversion
    (centre uniquely wins at (5,5), draws at (6,6), loses to a P1 line from (7,7));
  * the A3 exactness lemma re-verified *for the variant* -- charity never fires
    for the single-hide ablations, so those floors are exact optima, not bounds;
  * the geometry: exactly 8 of the full board's 49 winning lines can fire;
  * recorded-JSON guards on the shipped results file.
"""

import json
from pathlib import Path

import pytest

from collapse3.game import placement_pegs
from experiments.threepeg_floor import (
    EXPECTED_FLOORS,
    EXPECTED_OPENINGS,
    PLACEMENT_ROW,
    analyse_size,
    opening_values,
    winning_line_census,
)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "threepeg_floor_latest.json"
SIZES = tuple(range(2, 15))
_FLOOR_TOL = 5e-5


@pytest.fixture(scope="module")
def sweep():
    """Enumerate + floor every size once, under the three-peg restriction."""
    out = {}
    with placement_pegs(PLACEMENT_ROW):
        for r in SIZES:
            out[r] = analyse_size(r)
    return out  # global restored on context exit -- no leak into other tests


@pytest.fixture(scope="module")
def openings():
    out = {}
    with placement_pegs(PLACEMENT_ROW):
        for r in range(1, 15):
            out[r] = opening_values(r)
    return out


def test_eight_of_49_winning_lines_live():
    # Sibling geometry: beads live only on pegs (0,1,2), so only the row's lines
    # can ever fire -- 3 verticals + the row at 3 levels + 2 staircases = 8/49.
    assert winning_line_census() == {
        "total": 49, "live": 8, "verticals": 3, "row_levels": 3, "staircases": 2}


def test_variant_is_a_pure_placement_restriction():
    from collapse3.game import empty_state, get_legal_moves
    root = empty_state(4, 4)
    assert {m for m in get_legal_moves(root)} == {("place", p) for p in range(9)}
    with placement_pegs(PLACEMENT_ROW):
        assert {m for m in get_legal_moves(root)} == {("place", p) for p in PLACEMENT_ROW}
    # Restriction is scoped: the default (all 9 pegs) is restored on exit.
    assert {m for m in get_legal_moves(root)} == {("place", p) for p in range(9)}


def test_reproduce_gate_floors_all_sizes(sweep):
    for r, exp in EXPECTED_FLOORS.items():
        info = sweep[r]
        assert info["states"] == exp[0], f"state count drift at ({r},{r})"
        assert info["decisions"] == exp[1], f"decision count drift at ({r},{r})"
        assert abs(info["rows"]["hide_cooldown"]["floor"] - exp[2]) < _FLOOR_TOL
        assert abs(info["rows"]["hide_reserves"]["floor"] - exp[3]) < _FLOOR_TOL


def test_reproduce_gate_openings_all_sizes(openings):
    for r, exp in EXPECTED_OPENINGS.items():
        root_v, place_v = openings[r]
        assert root_v == exp[0], f"root value drift at ({r},{r})"
        assert place_v == exp[1], f"opening placement drift at ({r},{r})"


def test_certified_centre_strategy_inversion(openings):
    # The phase boundary the sibling exists to expose, with exact ground truth.
    assert openings[5][1] == [0, 10, 0]          # centre uniquely wins at (5,5)
    assert openings[6][1] == [10, 0, 10]         # centre only draws at (6,6)
    for r in range(7, 15):
        assert openings[r][1] == [10, -100, 10]  # centre loses to a P1 line


def test_exactness_lemma_holds_for_variant_all_sizes(sweep):
    # A3: re-verify Finding 4's lemma *for this variant* -- charity never fires
    # for the single-hide ablations, so those floors are exact optima not bounds.
    for r in SIZES:
        rows = sweep[r]["rows"]
        assert rows["hide_cooldown"]["no_common_legal_action"] == 0, f"cd charity fired at ({r},{r})"
        assert rows["hide_reserves"]["no_common_legal_action"] == 0, f"res charity fired at ({r},{r})"
    # Contrast: hiding both fields gates legality, so charity DOES fire somewhere.
    assert any(sweep[r]["rows"]["hide_both"]["no_common_legal_action"] > 0 for r in SIZES)


def test_hide_cooldown_is_non_monotonic(sweep):
    cd = [sweep[r]["rows"]["hide_cooldown"]["floor"] for r in SIZES]
    peak = max(range(len(cd)), key=lambda i: cd[i])
    assert SIZES[peak] == 7                       # peak at (7,7)
    assert cd[-1] < cd[peak]                       # declines afterwards (non-monotonic)


def test_hide_reserves_increments_collapse(sweep):
    hr = [sweep[r]["rows"]["hide_reserves"]["floor"] for r in SIZES]
    incs = [hr[i + 1] - hr[i] for i in range(len(hr) - 1)]
    assert all(i > 0 for i in incs)               # still rising at (14,14)
    # ...but the increments shrink toward the asymptote (saturation).
    assert incs[-1] < 0.01 < max(incs)


def test_recorded_json_matches(sweep):
    data = json.loads(RESULTS.read_text())
    assert data["provenance"]["config"]["placement_pegs"] == [0, 1, 2]
    res = data["results"]
    assert res["variant"] == "Three-Peg Collapse"
    assert res["winning_line_census"] == {
        "total": 49, "live": 8, "verticals": 3, "row_levels": 3, "staircases": 2}
    assert res["exactness_lemma_holds"] is True
    by_size = res["by_size"]
    for r, exp in EXPECTED_FLOORS.items():
        row = by_size[f"{r}_{r}"]
        assert row["states"] == exp[0]
        assert row["decisions"] == exp[1]
        assert abs(row["rows"]["hide_cooldown"]["floor"] - exp[2]) < _FLOOR_TOL
        assert abs(row["rows"]["hide_reserves"]["floor"] - exp[3]) < _FLOOR_TOL
        assert row["rows"]["hide_cooldown"]["no_common_legal_action"] == 0
        assert row["rows"]["hide_reserves"]["no_common_legal_action"] == 0
