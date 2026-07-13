"""Aliasing regret-floor tests."""

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state


def test_full_observation_has_zero_floor():
    memo = solve_all(empty_state(2, 2))
    res = regret_floor(memo, OBSERVATIONS["full"])
    # Full state is never aliased -> a memoryless policy pays no floor.
    assert res.floor == 0.0
    assert res.aliased_groups == 0


def test_lossy_floors_are_bounded_and_nonnegative():
    memo = solve_all(empty_state(3, 3))
    for name in ("hide_cooldown", "hide_reserves", "hide_both"):
        res = regret_floor(memo, OBSERVATIONS[name])
        assert 0.0 <= res.floor <= 2.0            # wdl regret is at most 2/decision
        assert res.aliased_groups >= 0
        assert res.obs_groups >= res.aliased_groups


def test_p0_only_restriction_changes_census_size():
    memo = solve_all(empty_state(3, 3))
    all_dec = regret_floor(memo, OBSERVATIONS["hide_cooldown"], p0_only=False)
    p0_dec = regret_floor(memo, OBSERVATIONS["hide_cooldown"], p0_only=True)
    assert p0_dec.n_decisions < all_dec.n_decisions
