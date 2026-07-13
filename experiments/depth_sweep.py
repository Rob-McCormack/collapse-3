"""Competence-per-ply: how much lookahead does a strategy need?

This asks the "is there a basic strategy?" question directly, separating the two
variables a shallow policy can vary: **evaluation quality** and **lookahead
depth**. Every decision is graded against the exact oracle.

Policies:
  * shallow references (no search): `rand` (guess a legal move -- the correct
    baseline, NOT 50%) and `myopic` (naive own-alignment greed).
  * bead-count ladder: `NPlyAgent` at increasing depth. Its leaf eval is the
    material differential, which is *provably flat* across all one-ply children
    (every legal move shifts it by exactly +1), so depth-1 is equivalent to
    random. Holds a (degenerate-at-1-ply) eval fixed while varying depth.
  * positional ladder: `HeuristicNPlyAgent` at increasing depth -- the same
    *functioning* `board_heuristic` at every depth, isolating lookahead from
    eval quality.

Regret is reported in win/draw/loss units (0-2), the same units as the aliasing
floors, so the two are directly comparable.

Run:
  python -m experiments.depth_sweep
  python -m experiments.depth_sweep --depths 1 2 3 4 5 6 --heur-depths 1 2 3
"""

import argparse
import sys
import time

from collapse3.agents import (
    HeuristicNPlyAgent,
    MyopicAgent,
    NPlyAgent,
    OptimalAgent,
    RandomAgent,
)
from collapse3.metrics import evaluate_competence
from experiments._provenance import announce, write_result

NAME = "depth_sweep"


def _row(report, opp_name, label, secs):
    return {
        "opponent": opp_name,
        "policy": label,
        "win_rate": report.win_rate,
        "draw_rate": report.draw_rate,
        "loss_rate": report.loss_rate,
        "decisions": report.decisions,
        "critical_decisions": report.critical_decisions,
        "optimal_rate_overall": report.optimal_rate,
        "optimal_rate_critical": report.optimal_rate_critical,
        "mean_regret_overall": report.mean_regret,
        "mean_regret_critical": report.mean_regret_critical,
        "mean_wdl_regret_overall": report.mean_wdl_regret,
        "mean_wdl_regret_critical": report.mean_wdl_regret_critical,
        "class_regret_hist": report.class_regret_hist,
        "outcomes": report.outcomes,
        "seconds": round(secs, 1),
    }


def _print_row(label, report, secs):
    print(f"  {label:>14} | {report.win_rate:>5.2f} {report.draw_rate:>5.2f} "
          f"{report.loss_rate:>5.2f} | {report.optimal_rate:>8.4f} "
          f"{report.optimal_rate_critical:>9.4f} | "
          f"{report.mean_wdl_regret_critical:>10.4f}   [{secs:.0f}s]")


def main(depths=(1, 2, 3, 4, 5, 6), heur_depths=(1, 2, 3), n_games=200,
         base_seed=12345, seat=0, res0=4, res1=4):
    opponents = {
        "optimal": lambda s: OptimalAgent(seed=s),
        "random": lambda s: RandomAgent(seed=s),
    }

    config = {
        "reserves": [res0, res1],
        "n_games": n_games,
        "base_seed": base_seed,
        "test_seat": seat,
        "depths": list(depths),
        "heur_depths": list(heur_depths),
        "opponents": list(opponents),
        "policies": "rand, myopic; bead-count NPlyAgent(depth); positional HeuristicNPlyAgent(depth)",
        "regret": "wdl units (0-2, comparable to aliasing floors) + raw solver-unit drop",
    }

    print(f"Competence-per-ply sweep  (reserves={res0},{res1}, seat=P{seat}, "
          f"{n_games} games/cell)")
    print("=" * 84)

    # (label, factory) in display order, per opponent.
    def policies():
        yield "rand", lambda s: RandomAgent(seed=s)
        yield "myopic", lambda s: MyopicAgent(seed=s)
        for d in depths:
            yield f"bead-{d}ply", (lambda s, d=d: NPlyAgent(depth=d, seed=s))
        for d in heur_depths:
            yield f"pos-{d}ply", (lambda s, d=d: HeuristicNPlyAgent(depth=d, seed=s))

    rows = []
    for opp_name, opp_factory in opponents.items():
        print(f"\nvs {opp_name}:")
        print(f"  {'policy':>14} | {'win':>5} {'draw':>5} {'loss':>5} | "
              f"{'opt(all)':>8} {'opt(crit)':>9} | {'wdlreg(crit)':>10}")
        print("  " + "-" * 74)
        for label, factory in policies():
            t0 = time.time()
            report = evaluate_competence(
                test_factory=factory, opp_factory=opp_factory,
                res0=res0, res1=res1, n_games=n_games, base_seed=base_seed, test_seat=seat,
            )
            secs = time.time() - t0
            _print_row(label, report, secs)
            rows.append(_row(report, opp_name, label, secs))

    print("\n  opt(crit) = optimal-move rate on *critical* decisions (where a mistake is possible).")
    print("  wdlreg(crit) = mean win/draw/loss-unit regret on critical decisions (0-2), "
          "comparable to the aliasing floors.")
    print("  bead-1ply == rand (flat one-ply eval); the positional eval reaches optimal at 2-ply.\n")

    path = write_result(NAME, config, {"rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--heur-depths", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--seat", type=int, default=0, choices=(0, 1))
    ap.add_argument("--reserves", nargs=2, type=int, default=[4, 4])
    args = ap.parse_args(sys.argv[1:] or None)
    main(tuple(args.depths), tuple(args.heur_depths), args.games, args.seed,
         args.seat, args.reserves[0], args.reserves[1])
