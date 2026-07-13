"""Opening-value surprises: first-mover asymmetry and the (r,r) phase transition.

These lock in the exact claims documented in Finding 6.
"""

from collapse3.game import empty_state
from collapse3.solver import game_value, reset_search_state


def test_first_player_never_loses_small_grid():
    reset_search_state()
    for r0 in range(1, 5):
        for r1 in range(1, 5):
            assert game_value(empty_state(r0, r1, 0)) >= 0


def test_reserve_lead_pays_off_only_for_the_mover():
    reset_search_state()
    # Mover with a strict reserve lead wins (by attrition at small sizes)...
    assert game_value(empty_state(2, 1, 0)) == 10
    assert game_value(empty_state(3, 2, 0)) == 10
    # ...but the replier's material lead buys only a draw.
    assert game_value(empty_state(1, 2, 0)) == 0
    assert game_value(empty_state(1, 4, 0)) == 0


def test_equal_reserves_draw_below_the_transition():
    reset_search_state()
    for r in (1, 2, 3, 4, 5):
        assert game_value(empty_state(r, r, 0)) == 0


def test_phase_transition_at_six():
    # The equal game becomes a first-player line win once material is large enough.
    reset_search_state()
    assert game_value(empty_state(6, 6, 0)) == 100


def test_value_is_move_order_invariant():
    # A robustness probe for ordering-sensitive alpha-beta / TT bugs: the exact
    # value must not depend on move-exploration order (hardens the (6,6) claim).
    for split in [(6, 6), (6, 3), (3, 6), (4, 4)]:
        base = game_value(empty_state(*split, 0), fresh=True)
        for seed in (1, 2, 3, 4, 5):
            assert game_value(empty_state(*split, 0), order_seed=seed) == base
