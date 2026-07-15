"""Six-Peg Collapse sibling: reproduce gate, exactness lemma, and the boundary
shift that tests Finding 12(iv).

The two-row board is expensive (9.15M states at (7,7)), so the default suite runs
only the cheap parts: the geometry census, a small-size reproduce gate (2..4),
and guards on the shipped results JSON (no recompute). The heavy recompute of
(5,5)..(7,7) is gated behind ``COLLAPSE3_SLOW=1``.

The headline test is behavioural: every boundary signature the three-peg sibling
showed near (6,6)-(7,7) is *delayed* on the 18-cell board -- direct evidence that
board capacity, not the mechanics, sets the boundary (Finding 13).
"""

import json
import os
from pathlib import Path

import pytest

from collapse3.game import empty_state, get_legal_moves, placement_pegs
from experiments.sixpeg_floor import (
    EXPECTED_FLOORS,
    EXPECTED_OPENINGS,
    PLACEMENT_ROWS,
    analyse_size,
)
from experiments.threepeg_floor import EXPECTED_FLOORS as TP_FLOORS
from experiments.threepeg_floor import winning_line_census

RESULTS = Path(__file__).resolve().parent.parent / "results" / "sixpeg_floor_latest.json"
_FLOOR_TOL = 5e-5


# --- cheap: geometry + engine restriction --------------------------------------
def test_sixteen_of_49_winning_lines_live():
    # Two rows (pegs 0-5): 6 verticals + 2 row-lines at 3 levels + 4 staircases.
    assert winning_line_census(PLACEMENT_ROWS) == {
        "total": 49, "live": 16, "verticals": 6, "row_levels": 6, "staircases": 4}


def test_variant_is_a_pure_placement_restriction():
    root = empty_state(4, 4)
    assert {m for m in get_legal_moves(root)} == {("place", p) for p in range(9)}
    with placement_pegs(PLACEMENT_ROWS):
        assert {m for m in get_legal_moves(root)} == {("place", p) for p in PLACEMENT_ROWS}
    assert {m for m in get_legal_moves(root)} == {("place", p) for p in range(9)}  # restored


# --- cheap: small-size reproduce gate (recompute (2,3,4), ~10s) -----------------
def test_reproduce_gate_small_sizes():
    with placement_pegs(PLACEMENT_ROWS):
        for r in (2, 3, 4):
            info = analyse_size(r)
            exp = EXPECTED_FLOORS[r]
            assert info["states"] == exp[0], f"state drift at ({r},{r})"
            assert info["decisions"] == exp[1], f"decision drift at ({r},{r})"
            assert abs(info["rows"]["hide_cooldown"]["floor"] - exp[2]) < _FLOOR_TOL
            assert abs(info["rows"]["hide_reserves"]["floor"] - exp[3]) < _FLOOR_TOL
            # exactness lemma at the small sizes
            assert info["rows"]["hide_cooldown"]["no_common_legal_action"] == 0
            assert info["rows"]["hide_reserves"]["no_common_legal_action"] == 0
            # openings gate
            exp_o = EXPECTED_OPENINGS[r]
            assert info["root"] == exp_o[0] and info["placements"] == exp_o[1]


# --- cheap: guards on the shipped record (no recompute) -------------------------
@pytest.fixture(scope="module")
def record():
    return json.loads(RESULTS.read_text())


def test_recorded_json_matches(record):
    assert record["provenance"]["config"]["placement_pegs"] == [0, 1, 2, 3, 4, 5]
    res = record["results"]
    assert res["variant"] == "Six-Peg Collapse"
    assert res["winning_line_census"] == {
        "total": 49, "live": 16, "verticals": 6, "row_levels": 6, "staircases": 4}
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


def test_recorded_reaches_seven_seven(record):
    # (7,7) is the point that makes the finding: it must be in the shipped record.
    assert "7_7" in record["results"]["by_size"]
    assert record["results"]["by_size"]["7_7"]["states"] == 9152201


# --- cheap: the finding's behavioural claim (Finding 13) ------------------------
def test_boundary_is_delayed_versus_three_peg(record):
    """Every boundary signature is later on the 18-cell board than the 9-cell one."""
    by = record["results"]["by_size"]

    # (i) Cooldown: three-peg PEAKS at (7,7); six-peg is still rising there, lower.
    assert by["7_7"]["hide_cooldown_increment"] > 0            # still climbing
    assert by["7_7"]["rows"]["hide_cooldown"]["floor"] < TP_FLOORS[7][2]  # below 3-peg's peak
    tp_cd_still_rising_at_7 = TP_FLOORS[7][2] - TP_FLOORS[6][2]  # +0.0002: three-peg has turned over
    assert by["7_7"]["hide_cooldown_increment"] > tp_cd_still_rising_at_7

    # (ii) Reserves: three-peg's increments collapse after (5,5); six-peg's barely roll off.
    tp_res_inc_7 = TP_FLOORS[7][3] - TP_FLOORS[6][3]           # +0.0262 (well into collapse)
    assert by["7_7"]["hide_reserves_increment"] > tp_res_inc_7  # slower collapse -> later saturation

    # (iii) Openings: the root turns into a P0 win ~3 sizes later than on three pegs.
    openings = record["results"]["openings"]
    assert openings["3_3"]["root"] == 0                        # still a draw where three-peg already won
    assert openings["5_5"]["root"] == 0
    assert openings["6_6"]["root"] == 10                       # only now a P0 attrition win
    # ...and the centre-uniquely-wins phase lands at (7,7), where three-peg was at (5,5).
    centre_7 = openings["7_7"]["placements"][1]["value"]       # peg 1
    end_7 = openings["7_7"]["placements"][0]["value"]          # peg 0
    assert centre_7 == 10 and end_7 == 0


# --- slow: full recompute behind a flag ----------------------------------------
@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to recompute (5,5)-(6,6) (~5 min)",
)
def test_recompute_gate_five_six():
    with placement_pegs(PLACEMENT_ROWS):
        for r in (5, 6):
            info = analyse_size(r)
            exp = EXPECTED_FLOORS[r]
            assert info["states"] == exp[0]
            assert info["decisions"] == exp[1]
            assert abs(info["rows"]["hide_cooldown"]["floor"] - exp[2]) < _FLOOR_TOL
            assert abs(info["rows"]["hide_reserves"]["floor"] - exp[3]) < _FLOOR_TOL


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to recompute (7,7) (~20 min, ~10GB RAM)",
)
def test_recompute_seven_seven_matches_record():
    with placement_pegs(PLACEMENT_ROWS):
        info = analyse_size(7)
    exp = EXPECTED_FLOORS[7]
    assert info["states"] == exp[0]
    assert info["decisions"] == exp[1]
    assert abs(info["rows"]["hide_cooldown"]["floor"] - exp[2]) < _FLOOR_TOL
    assert abs(info["rows"]["hide_reserves"]["floor"] - exp[3]) < _FLOOR_TOL
    assert info["rows"]["hide_cooldown"]["no_common_legal_action"] == 0
    assert info["rows"]["hide_reserves"]["no_common_legal_action"] == 0
