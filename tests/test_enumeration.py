"""Enumeration tests: agreement with the folded solver + value invariants."""

import random

from collapse3.enumeration import reachable_states, solve_all, wdl
from collapse3.game import empty_state
from collapse3.solver import game_value, reset_search_state

VALID_VALUES = {-100, -10, 0, 10, 100}


def test_root_value_matches_folded_solver():
    for r in (1, 2, 3):
        reset_search_state()
        root = empty_state(r, r)
        memo = solve_all(root)
        assert memo[root] == game_value(root)


def test_all_values_are_valid_terminals():
    memo = solve_all(empty_state(2, 2))
    assert set(memo.values()) <= VALID_VALUES
    assert len(memo) == len(reachable_states(empty_state(2, 2)))


def test_enumeration_agrees_with_solver_on_sampled_states():
    reset_search_state()
    root = empty_state(3, 3)
    memo = solve_all(root)
    rng = random.Random(0)
    for s in rng.sample(list(memo), min(200, len(memo))):
        assert memo[s] == game_value(s)


def test_wdl_units():
    assert wdl(100, 0) == 1 and wdl(100, 1) == -1
    assert wdl(-10, 0) == -1 and wdl(-10, 1) == 1
    assert wdl(0, 0) == 0
