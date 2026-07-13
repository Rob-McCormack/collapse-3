"""Metrics / harness tests: determinism and optimal-agent invariants."""

from collapse3.agents import NPlyAgent, OptimalAgent, RandomAgent
from collapse3.metrics import evaluate_competence


def test_runs_are_reproducible():
    kw = dict(res0=3, res1=3, n_games=20, base_seed=7, test_seat=0)
    a = evaluate_competence(lambda s: NPlyAgent(1, s), lambda s: RandomAgent(s), **kw)
    b = evaluate_competence(lambda s: NPlyAgent(1, s), lambda s: RandomAgent(s), **kw)
    assert a.outcomes == b.outcomes
    assert a.total_regret == b.total_regret
    assert a.optimal_decisions == b.optimal_decisions


def test_optimal_agent_has_zero_regret():
    report = evaluate_competence(
        test_factory=lambda s: OptimalAgent(seed=s),
        opp_factory=lambda s: RandomAgent(seed=s),
        res0=3, res1=3, n_games=25, base_seed=1, test_seat=0,
    )
    assert report.total_regret == 0
    assert report.optimal_rate == 1.0
    # Optimal play never loses a solved-draw game.
    assert report.loss_rate == 0.0


def test_aggregate_optimal_rate_exceeds_critical_only():
    report = evaluate_competence(
        test_factory=lambda s: NPlyAgent(1, s),
        opp_factory=lambda s: RandomAgent(s),
        res0=4, res1=4, n_games=40, base_seed=3, test_seat=0,
    )
    assert report.decisions > 0
    if report.critical_decisions > 0 and report.optimal_rate_critical < 1.0:
        # The flattering-aggregate effect: overall rate should be >= critical rate.
        assert report.optimal_rate >= report.optimal_rate_critical
