"""Optimal-move rate: aggregate vs critical-only.

Reproduces the "aggregate optimal-move rate is adversarially flattering" claim
(the "one-ply player gets ~79% of moves optimal" phenomenon) *reproducibly*.

For each opponent distribution we play a fixed, seeded set of games with a
one-ply searcher in the test seat and report, via the oracle:
    * overall optimal-move rate (every decision, most of which are non-critical),
    * critical-only optimal-move rate (decisions where a mistake is possible),
    * mean value-based regret (overall and critical).

The gap between the two optimal-move rates is the whole point: the aggregate
number is inflated by the many states where every move is optimal.

Run:  python -m experiments.optimal_move_rate
"""

from collapse3.agents import MyopicAgent, NPlyAgent, OptimalAgent, RandomAgent
from collapse3.metrics import evaluate_competence
from experiments._provenance import announce, write_result

NAME = "optimal_move_rate"


def main() -> None:
    res0 = res1 = 4
    n_games = 200
    base_seed = 12345
    depth = 1  # one-ply test agent

    opponents = {
        "random": lambda s: RandomAgent(seed=s),
        "myopic": lambda s: MyopicAgent(seed=s),
        "one_ply": lambda s: NPlyAgent(depth=1, seed=s),
        "optimal": lambda s: OptimalAgent(seed=s),
    }

    config = {
        "reserves": [res0, res1],
        "n_games": n_games,
        "base_seed": base_seed,
        "test_agent": f"{depth}-ply",
        "opponents": list(opponents),
        "regret": "value-based (oriented solver-unit drop; class_regret = ordinal ladder steps)",
    }

    print(f"One-ply optimal-move rate vs opponent pools "
          f"(reserves={res0},{res1}, {n_games} games/pool)")
    print("=" * 76)

    rows = []
    for opp_name, opp_factory in opponents.items():
        report = evaluate_competence(
            test_factory=lambda s: NPlyAgent(depth=depth, seed=s),
            opp_factory=opp_factory,
            res0=res0,
            res1=res1,
            n_games=n_games,
            base_seed=base_seed,
            test_seat=0,
        )
        print(f"\nvs {opp_name}:")
        print(report.summary())
        rows.append({
            "opponent": opp_name,
            "win_rate": report.win_rate,
            "draw_rate": report.draw_rate,
            "loss_rate": report.loss_rate,
            "decisions": report.decisions,
            "critical_decisions": report.critical_decisions,
            "optimal_rate_overall": report.optimal_rate,
            "optimal_rate_critical": report.optimal_rate_critical,
            "mean_regret_overall": report.mean_regret,
            "mean_regret_critical": report.mean_regret_critical,
            "class_regret_hist": report.class_regret_hist,
            "outcomes": report.outcomes,
        })

    path = write_result(NAME, config, {"rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    main()
