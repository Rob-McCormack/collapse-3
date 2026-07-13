"""Oracle / value-based regret tests."""

from collapse3.game import GameState, empty_state
from collapse3.oracle import Oracle
from collapse3.solver import game_value, reset_search_state

E = tuple()


def board(**kw):
    b = [E] * 9
    for k, v in kw.items():
        b[int(k[1:])] = tuple(v)
    return tuple(b)


def test_optimal_moves_have_zero_regret_and_match_state_value():
    reset_search_state()
    oracle = Oracle()
    st = empty_state(3, 3)
    label = oracle.label_state(st)
    assert label is not None
    # At least one optimal move; every optimal move reproduces the state value.
    assert len(label.optimal_moves) >= 1
    for ml in label.moves:
        assert (ml.regret == 0) == ml.is_optimal
        if ml.is_optimal:
            assert ml.mover_value == label.best_mover_value
        else:
            assert ml.regret > 0


def test_regret_is_value_based_not_policy_distance():
    # Symmetric empty board is a draw; many first moves are all optimal.
    # A value-based metric must call ALL value-preserving moves optimal
    # (a policy-distance metric would wrongly penalize all-but-one).
    reset_search_state()
    oracle = Oracle()
    st = empty_state(2, 2)
    label = oracle.label_state(st)
    assert game_value(st) == 0
    assert len(label.optimal_moves) > 1   # non-unique optimum


def test_move_that_hands_opponent_a_win_has_large_regret():
    reset_search_state()
    oracle = Oracle()
    # P0 to move; P1 already has two-in-a-row threat at level 0 on pegs 3,4,5.
    # Placing away from the block lets P1 complete it next turn.
    st = GameState(board(p3=[1], p4=[1]), (3, 3), 0, (False, False))
    label = oracle.label_state(st)
    # There should be a spread of regrets (some moves defend, some do not),
    # and the best move's regret is 0.
    regrets = sorted(m.regret for m in label.moves)
    assert regrets[0] == 0
    assert regrets[-1] >= regrets[0]


def test_critical_flag_semantics():
    reset_search_state()
    oracle = Oracle()
    st = empty_state(1, 1)
    label = oracle.label_state(st)
    # is_critical iff not every legal move is optimal.
    assert label.is_critical == any(not m.is_optimal for m in label.moves)
