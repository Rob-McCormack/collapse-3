"""Win rate vs. competence: performance is not competence.

Reproduces the "dominates the pool while still playing imperfectly" phenomenon
(the "200-0 but X% of moves wrong" observation) reproducibly. A test agent is
scored two ways against several opponent pools:

    performance  -- win/draw/loss rate (opponent-distribution dependent),
    competence   -- value-based regret & optimal-move rate (oracle ground truth).

A high win rate coexisting with nonzero regret is the crux: the win rate reflects
the opponent pool, not the agent's distance from optimal play.

Run:  python -m experiments.win_rate_vs_regret
"""

from collapse3.agents import MyopicAgent, NPlyAgent, RandomAgent
from collapse3.metrics import evaluate_competence
from experiments._provenance import announce, write_result

NAME = "win_rate_vs_regret"


def main() -> None:
    res0 = res1 = 4
    n_games = 200
    base_seed = 999

    # A 2-ply searcher: strong enough to dominate weak pools, still imperfect.
    def test_factory(s):
        return NPlyAgent(depth=2, seed=s)

    opponents = {
        "random": lambda s: RandomAgent(seed=s),
        "myopic": lambda s: MyopicAgent(seed=s),
        "one_ply": lambda s: NPlyAgent(depth=1, seed=s),
    }

    config = {
        "reserves": [res0, res1],
        "n_games": n_games,
        "base_seed": base_seed,
        "test_agent": "2-ply",
        "opponents": list(opponents),
    }

    print(f"2-ply agent: performance vs competence (reserves={res0},{res1}, "
          f"{n_games} games/pool)")
    print("=" * 76)

    rows = []
    for opp_name, opp_factory in opponents.items():
        report = evaluate_competence(
            test_factory=test_factory,
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
            "optimal_rate_overall": report.optimal_rate,
            "optimal_rate_critical": report.optimal_rate_critical,
            "mean_regret_overall": report.mean_regret,
            "mean_regret_critical": report.mean_regret_critical,
            "outcomes": report.outcomes,
        })

    path = write_result(NAME, config, {"rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    main()
