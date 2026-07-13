"""Memory-floor tests: destroyed counts reconstruct reserves; enumerated floor."""

from collapse3.aliasing import OBSERVATIONS, memory_reserves_obs, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state
from collapse3.learning import destroyed_from_state, memory_reserves_obs as learn_mem_obs


def test_destroyed_counts_reconstruct_reserves():
    root = empty_state(3, 3)
    memo = solve_all(root)
    for s in list(memo)[:500]:
        d = destroyed_from_state(s, (3, 3))
        on0 = sum(col.count(0) for col in s.board)
        on1 = sum(col.count(1) for col in s.board)
        assert s.res[0] == 3 - on0 - d[0]
        assert s.res[1] == 3 - on1 - d[1]


def test_memory_augmented_observation_has_zero_floor():
    root = empty_state(3, 3)
    memo = solve_all(root)
    blind = regret_floor(memo, OBSERVATIONS["hide_reserves"], p0_only=True)
    memory = regret_floor(memo, memory_reserves_obs((3, 3)), p0_only=True)
    assert blind.floor > 0
    assert memory.floor == 0.0


def test_training_and_audit_observations_agree_on_key():
    s = empty_state(2, 2)
    mem = (1, 0)
    assert learn_mem_obs((2, 2))(s, mem) == (s.board, s.turn, s.cooldown, mem)
