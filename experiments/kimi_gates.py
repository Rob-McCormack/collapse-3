"""Gates A and B for the KimiAgent arc: distributional evaluation of v1/v2.

Gate A (exact line): kimi-v1 vs a deterministic optimal opponent at (4,4),
Kimi as P0 -- the game should be a draw with zero regret on the single line
(repeated across tie-break seeds).

Gate B (interval): kimi-v1 vs an eps-optimal opponent at (4,4), Kimi as P0,
3 seeds x 400 games per epsilon in {0.05, 0.25}. v2 runs the SAME paired
seeds for comparison. NOTE: the originating session reported 0 losses in
1,200 games at "25% noise" under its own opponent implementation; with this
repo's eps-optimal opponent (a uniformly random legal move with probability
eps per decision) v1 records 9 losses / 1,200 at eps=0.25 and 1 / 1,200 at
eps=0.05 -- and the losses travel down the certified Gate C exploit family
(unblocked towers), so the distributional test only surfaces the flaw when
noise happens to walk the exploit line.

The point of recording Gate B at all is its juxtaposition with Gate C
(experiments/best_response.py): the same policy that survives 1,200
noisy-adversarial games with zero losses carries a 5-ply forced refutation.

Run:  python -m experiments.kimi_gates
"""

from typing import Dict

from collapse3.agents import KimiAgentV1, KimiAgentV2, OptimalAgent
from collapse3.metrics import evaluate_competence
from collapse3.oracle import Oracle
from experiments._provenance import announce, write_result

NAME = "kimi_gates"

RES = 4
SEEDS = (0, 1, 2)
GAMES_PER_SEED = 400
EPSILONS = (0.05, 0.25)


def binomial_upper_95(losses: int, n: int) -> float:
    """Exact Clopper-Pearson 95% upper bound; closed form for 0 losses."""
    if losses == 0:
        return 1.0 - 0.05 ** (1.0 / n)
    # Conservative fallback (not needed for the recorded result).
    from math import sqrt
    p = losses / n
    return min(1.0, p + 1.96 * sqrt(p * (1 - p) / n) + 1 / n)


def gate_a(oracle: Oracle) -> Dict[str, object]:
    per_seed = []
    for seed in SEEDS:
        rep = evaluate_competence(
            test_factory=lambda s: KimiAgentV1(),
            opp_factory=lambda s, _seed=seed: OptimalAgent(seed=_seed, epsilon=0.0),
            res0=RES, res1=RES, n_games=1, base_seed=seed, test_seat=0,
            oracle=oracle,
        )
        per_seed.append({
            "opponent_seed": seed,
            "outcome": dict(rep.outcomes),
            "kimi_total_regret": rep.total_regret,
        })
    ok = all(s["outcome"] == {"draw": 1} and s["kimi_total_regret"] == 0
             for s in per_seed)
    return {"pass": ok, "lines": per_seed}


def gate_b(oracle: Oracle) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for eps in EPSILONS:
        for name, factory in (("kimi-v1", KimiAgentV1), ("kimi-v2", KimiAgentV2)):
            losses = wins = draws = 0
            crit = crit_opt = crit_wdl = 0
            per_seed = []
            for seed in SEEDS:
                rep = evaluate_competence(
                    test_factory=lambda s: factory(),
                    opp_factory=lambda s, _e=eps: OptimalAgent(seed=s, epsilon=_e),
                    res0=RES, res1=RES, n_games=GAMES_PER_SEED,
                    base_seed=seed * 100_000, test_seat=0, oracle=oracle,
                )
                losses += rep.outcomes.get("true loss", 0) + rep.outcomes.get("attrition loss", 0)
                wins += rep.outcomes.get("true win", 0) + rep.outcomes.get("attrition win", 0)
                draws += rep.outcomes.get("draw", 0)
                crit += rep.critical_decisions
                crit_opt += rep.optimal_on_critical
                crit_wdl += rep.total_wdl_regret_critical
                per_seed.append({"seed": seed, "outcomes": dict(rep.outcomes),
                                 "crit_opt_rate": round(rep.optimal_rate_critical, 4),
                                 "crit_wdl_regret": round(rep.mean_wdl_regret_critical, 4)})
            n = len(SEEDS) * GAMES_PER_SEED
            out[f"{name}@eps={eps}"] = {
                "games": n, "epsilon": eps,
                "wins": wins, "draws": draws, "losses": losses,
                "loss_upper_95": round(binomial_upper_95(losses, n), 5),
                "crit_opt_rate": round(crit_opt / crit, 4) if crit else None,
                "crit_wdl_regret": round(crit_wdl / crit, 4) if crit else None,
                "per_seed": per_seed,
            }
    return out


def main() -> None:
    oracle = Oracle()
    print(f"Gate A: kimi-v1 vs deterministic optimal, ({RES},{RES}), Kimi P0")
    a = gate_a(oracle)
    print(f"  pass={a['pass']}  {a['lines']}")

    print(f"\nGate B: vs eps-optimal, ({RES},{RES}), Kimi P0, "
          f"{len(SEEDS)} seeds x {GAMES_PER_SEED} games per epsilon")
    b = gate_b(oracle)
    for name, row in b.items():
        print(f"  {name}: W/D/L {row['wins']}/{row['draws']}/{row['losses']}"
              f"  loss upper-95 {row['loss_upper_95']:.5f}"
              f"  crit-opt {row['crit_opt_rate']}"
              f"  crit-wdl-regret {row['crit_wdl_regret']}")

    path = write_result(NAME, {
        "reserves": [RES, RES], "seeds": list(SEEDS),
        "games_per_seed": GAMES_PER_SEED, "epsilons": list(EPSILONS),
    }, {"gate_a": a, "gate_b": b})
    announce(NAME, path)


if __name__ == "__main__":
    main()
