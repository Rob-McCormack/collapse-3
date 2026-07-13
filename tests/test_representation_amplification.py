"""Guard the budget-sweep record (the corrected 'undertraining' bullet).

History: the docs once cited a 1x-8x budget sweep as showing "the gap never
closes" while the shipped results file contained NO sweep (``curve: []``).
External review caught it; the sweep was then actually run and showed the
opposite surface behaviour -- on-policy regret FALLING with budget -- which is
consistent with the floors: the trained agent's interface (mask-aware
hide_cooldown at (4,4)) has an exact floor of 0, so scaling converges to it.
These tests make both mistakes structurally unrepeatable: the sweep must exist
in the shipped JSON, and its shape must match what the docs now claim.
"""

import json
from pathlib import Path

RESULTS = (Path(__file__).resolve().parent.parent
           / "results" / "representation_amplification_latest.json")


def _payload():
    return json.loads(RESULTS.read_text())["results"]


def test_recorded_curve_exists_and_covers_1x_to_8x():
    curve = _payload()["curve"]
    assert [row["multiplier"] for row in curve] == [1, 2, 4, 8]
    assert curve[0]["episodes"] == 60_000 and curve[-1]["episodes"] == 480_000


def test_recorded_regret_falls_with_budget_toward_the_interface_floor():
    curve = _payload()["curve"]
    regrets = [row["mean_regret"] for row in curve]
    assert regrets[0] > regrets[-1]                  # scaling helps on-policy...
    assert regrets == sorted(regrets, reverse=True)  # ...monotonically here
    assert regrets[-1] <= 0.001                      # ...down to ~the 0 floor
    # The matched (visited-state) floor is ~0 throughout: the agent's own
    # interface carries no aliasing cost, so this sweep CANNOT exhibit the wall.
    assert all(row["matched_floor"] <= 0.0005 for row in curve)
