"""Action-mask leakage: the aliasing floor is a property of the interface.

External review showed the published floors price a mask-blind interface, while
the repo's trained agents also see the legal-move list -- which leaks hidden
state. These tests lock the mask-aware floors (Finding 4's interface caveat):

  * hide_cooldown's floor is PURE mask leakage: cooldown matters only through
    removal legality, so the mask reconstructs it and the floor is exactly 0;
  * hide_reserves keeps a real residual at (4,4) (mask reveals zero-vs-positive
    reserve, not the count) -- ~30x smaller than the mask-blind 0.0805;
  * masked groups share one legal-move set, so the charity rule never fires and
    monotonicity (hide_both >= hide_reserves) is restored.
"""

import json
from pathlib import Path

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state, get_legal_moves

RESULTS = Path(__file__).resolve().parent.parent / "results" / "masked_floor_latest.json"


def _with_mask(base):
    return lambda s: (base(s), tuple(sorted(get_legal_moves(s))))


def test_mask_aware_floors_vanish_at_3_3():
    memo = solve_all(empty_state(3, 3))
    for name in ("hide_cooldown", "hide_reserves", "hide_both"):
        masked = regret_floor(memo, _with_mask(OBSERVATIONS[name]))
        assert masked.floor == 0.0
        assert masked.no_common_legal_action == 0   # charity can't fire under a mask


def test_recorded_masked_floors():
    by_size = json.loads(RESULTS.read_text())["results"]["by_size"]
    d44, d55 = by_size["4_4"]["rows"], by_size["5_5"]["rows"]
    # Cooldown floor is (almost) pure mask leakage: 0 at (4,4), trace at (5,5).
    assert d44["hide_cooldown"]["floor_mask_aware"] == 0.0
    assert d55["hide_cooldown"]["floor_mask_aware"] < 1e-4
    # Reserves floor survives the mask but collapses ~30-70x from the blind one.
    assert abs(d44["hide_reserves"]["floor_mask_blind"] - 0.0805) < 5e-4
    assert abs(d44["hide_reserves"]["floor_mask_aware"] - 0.0026) < 5e-4
    assert abs(d55["hide_reserves"]["floor_mask_blind"] - 0.1677) < 5e-4
    assert abs(d55["hide_reserves"]["floor_mask_aware"] - 0.0024) < 5e-4
    # Under a mask, monotonicity is restored and charity never fires.
    for rows in (d44, d55):
        assert (rows["hide_both"]["floor_mask_aware"]
                >= rows["hide_reserves"]["floor_mask_aware"])
        for row in rows.values():
            assert row["masked_no_common_legal_action"] == 0
