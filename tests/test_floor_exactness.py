"""Exactness of the single-hide aliasing floors (Finding 4).

The floor's "charity rule" (an observation group with no action legal in *every*
member contributes 0) keeps every floor a valid lower bound. For the single-hide
ablations we claim something stronger: the charity rule **never fires**, so those
floors are the *exact* optima of the best memoryless policy, not conservative
bounds. This test scans every decision-state group and enforces it -- if the
structural argument in the FINDINGS lemma has a hole, this catches it.

`hide_both` is the deliberate contrast: hiding reserves *and* cooldown makes
action legality itself diverge across aliased states, so the charity rule fires
and the column is a deflated lower bound (hence its non-monotonicity).
"""

import json
import os
from pathlib import Path

import pytest

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state

SINGLE_HIDE = ("hide_cooldown", "hide_reserves")
RESULTS = Path(__file__).resolve().parent.parent / "results" / "aliasing_floor_latest.json"


def _floors(r0, r1):
    memo = solve_all(empty_state(r0, r1))
    return {name: regret_floor(memo, OBSERVATIONS[name])
            for name in ("full", *SINGLE_HIDE, "hide_both")}


def test_single_hide_floors_are_exact_at_3_3():
    f = _floors(3, 3)
    # Every single-hide group has a common legal action -> floor is exact.
    for name in SINGLE_HIDE:
        assert f[name].no_common_legal_action == 0
    # Contrast: hiding both fields gates legality, so the charity rule DOES fire.
    assert f["hide_both"].no_common_legal_action > 0


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to scan the full (4,4) census (~70s)",
)
def test_single_hide_floors_are_exact_at_4_4():
    f = _floors(4, 4)
    for name in SINGLE_HIDE:
        assert f[name].no_common_legal_action == 0
    assert abs(f["hide_reserves"].floor - 0.0805) < 5e-4
    assert f["hide_both"].no_common_legal_action > 0


def test_recorded_5_5_census_is_exact():
    """Guard the load-bearing (5,5) floor's exactness without re-enumerating.

    Re-solving (5,5) is too heavy for CI (12.7M states), so the census scan is
    run by ``experiments/aliasing_floor.py`` and recorded in the results JSON
    that ships in the bundle. This asserts the shipped numbers show the charity
    rule never fired for the single-hides -- i.e. the 0.1677 floor is exact.
    """
    data = json.loads(RESULTS.read_text())
    assert data["provenance"]["config"]["reserves"] == [5, 5]
    rows = data["results"]["rows"]
    for name in SINGLE_HIDE:
        assert rows[name]["no_common_legal_action"] == 0
    assert abs(rows["hide_reserves"]["floor_all_decisions"] - 0.1677) < 5e-4
    assert rows["hide_both"]["no_common_legal_action"] > 0
