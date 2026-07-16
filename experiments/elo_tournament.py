"""The Elo mirage, measured exactly: ratings vs regret vs exploitability.

A full round-robin at (4,4) among the repo's named agents, every game graded
by the oracle. Three columns for each agent, from the SAME games:

  * Elo        -- maximum-likelihood (Bradley-Terry) rating fitted to the
                  round-robin outcomes, draws scored 1/2. Pure performance:
                  a function of wins/draws/losses only.
  * regret     -- mean WDL-unit regret per decision over the tournament's own
                  state distribution (oracle-graded). Competence.
  * worst case -- the exact best-response certificate (Finding 8's Gate C)
                  for the deterministic policies; "n/a" for stochastic ones.

The question the table answers: does a win-rate-derived rating separate agents
the way ground truth does? The prediction from Findings 1/2/8 is that it
cannot at the top -- the equal game is a forced draw, so once an agent stops
losing, outcomes (and therefore Elo) saturate, while regret and exploitability
keep separating. This experiment turns that prediction into measured numbers.

Method notes (fitting):
  * Ratings are fitted by minorization-maximization on the logistic model
    E = 1 / (1 + 10^((Rb-Ra)/400)); each agent's expected total equals its
    actual total. Order of games is irrelevant (unlike online Elo updates).
  * Regularization: every agent gets 2 virtual draws against a fixed
    1500-rated anchor, so an unbeaten agent has a finite rating. This is the
    only prior; it also pins the scale (ratings are otherwise
    translation-invariant).

Run:  python -m experiments.elo_tournament [base_seed]
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from collapse3.agents import (
    HeuristicNPlyAgent,
    HeuristicOnePlyAgent,
    KimiAgentV1,
    KimiAgentV2,
    MyopicAgent,
    NPlyAgent,
    OptimalAgent,
    RandomAgent,
)
from collapse3.game import empty_state, orient
from collapse3.metrics import play_game
from collapse3.oracle import Oracle
from collapse3.solver import game_value
from experiments._provenance import announce, write_result

NAME = "elo_tournament"

RES = 4
GAMES_PER_ORDERED_PAIR = 50
ANCHOR_RATING = 1500.0
ANCHOR_GAMES = 2.0          # virtual draws vs the anchor (regularization)

ROSTER = {
    "random": lambda seed: RandomAgent(seed),
    "myopic": lambda seed: MyopicAgent(seed),
    "bead-1ply": lambda seed: NPlyAgent(1, seed),
    "pos-1ply": lambda seed: HeuristicNPlyAgent(1, seed),
    "one-ply-heur": lambda seed: HeuristicOnePlyAgent(seed),
    "pos-2ply": lambda seed: HeuristicNPlyAgent(2, seed),
    "kimi-v1": lambda seed: KimiAgentV1(),
    "kimi-v2": lambda seed: KimiAgentV2(),
    "optimal": lambda seed: OptimalAgent(seed),
}

# Exact worst-case certificates (Finding 8 / experiments/best_response.py).
# Only deterministic policies have one; OptimalAgent's tie-breaks are seeded
# but every optimal move preserves value, so >= draw is a theorem for it.
WORST_CASE = {
    "kimi-v1": "forced loss (5-6 plies)",
    "kimi-v2": "forced loss (7-8 plies)",
    "optimal": ">= draw (theorem)",
}


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def fit_elo(scores: Dict[str, float], games: Dict[Tuple[str, str], int],
            iters: int = 400) -> Dict[str, float]:
    """MLE Bradley-Terry fit: per agent, expected total == actual total."""
    names = sorted(scores)
    r = {n: ANCHOR_RATING for n in names}

    def total_expected(n: str, rn: float) -> float:
        t = ANCHOR_GAMES * expected(rn, ANCHOR_RATING)
        for m in names:
            if m != n:
                t += games.get((n, m), 0) * expected(rn, r[m])
        return t

    for _ in range(iters):
        for n in names:
            target = scores[n] + ANCHOR_GAMES * 0.5
            lo, hi = -2000.0, 6000.0
            for _ in range(60):                      # bisection (monotone)
                mid = (lo + hi) / 2
                if total_expected(n, mid) < target:
                    lo = mid
                else:
                    hi = mid
            r[n] = (lo + hi) / 2
    return r


def scores_from_h2h(h2h: Dict[str, Dict[str, Dict[str, int]]]):
    """Recover (score, n_games) from the recorded undirected head-to-head table.

    ``h2h[a][b]`` aggregates both seat orders (100 games per unordered pair).
    """
    names = sorted(h2h)
    score: Dict[str, float] = {n: 0.0 for n in names}
    n_games: Dict[Tuple[str, str], int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            w, d, l = h2h[a][b]["w"], h2h[a][b]["d"], h2h[a][b]["l"]
            n = w + d + l
            score[a] += w + 0.5 * d
            score[b] += l + 0.5 * d
            n_games[(a, b)] = n
            n_games[(b, a)] = n
    return score, n_games


def bootstrap_elo_gap(
    h2h: Dict[str, Dict[str, Dict[str, int]]],
    a: str = "kimi-v2",
    b: str = "optimal",
    n_boot: int = 1000,
    seed: int = 0,
    fit_iters: int = 80,
) -> Dict[str, object]:
    """Nonparametric bootstrap CI for ``elo[a] - elo[b]`` from recorded H2H.

    Resamples each unordered pair's 100 outcomes from its empirical multinomial
    (with replacement), refits Bradley-Terry, and reports the percentile CI.
    Cheap (~1 min) — does not replay games.
    """
    names = sorted(h2h)
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
    gaps: List[float] = []
    for i in range(n_boot):
        rng = random.Random(seed + i)
        score = {n: 0.0 for n in names}
        n_games: Dict[Tuple[str, str], int] = {}
        for x, y in pairs:
            w, d, l = h2h[x][y]["w"], h2h[x][y]["d"], h2h[x][y]["l"]
            n = w + d + l
            outcomes = [0] * w + [1] * d + [2] * l
            samp = rng.choices(outcomes, k=n)
            sw, sd, sl = samp.count(0), samp.count(1), samp.count(2)
            score[x] += sw + 0.5 * sd
            score[y] += sl + 0.5 * sd
            n_games[(x, y)] = n
            n_games[(y, x)] = n
        ratings = fit_elo(score, n_games, iters=fit_iters)
        gaps.append(ratings[a] - ratings[b])
    gaps.sort()
    lo = gaps[int(0.025 * n_boot)]
    hi = gaps[int(0.975 * n_boot)]
    return {
        "pair": [a, b],
        "n_boot": n_boot,
        "seed": seed,
        "fit_iters": fit_iters,
        "point_gap": round(sum(gaps) / n_boot, 1),  # bootstrap mean (approx)
        "median_gap": round(gaps[n_boot // 2], 1),
        "ci95": [round(lo, 1), round(hi, 1)],
        "p_positive": round(sum(g > 0 for g in gaps) / n_boot, 4),
        "method": ("nonparametric bootstrap over unordered-pair outcome "
                   "multinomials from the recorded head-to-head; Bradley-Terry "
                   "refit each replicate"),
    }


def bootstrap_recorded(path: Path = None) -> Dict[str, object]:
    """Attach a bootstrap CI to the shipped tournament JSON (no game replay)."""
    path = path or (Path(__file__).resolve().parent.parent
                    / "results" / "elo_tournament_latest.json")
    data = json.loads(path.read_text())
    boot = bootstrap_elo_gap(data["results"]["head_to_head"])
    # Prefer the exact recorded point gap over the bootstrap mean.
    table = {row["agent"]: row for row in data["results"]["table"]}
    boot["point_gap"] = round(table["kimi-v2"]["elo"] - table["optimal"]["elo"], 1)
    data["results"]["bootstrap_kimi_v2_minus_optimal"] = boot
    path.write_text(json.dumps(data, indent=2) + "\n")
    # Keep the timestamped sibling in sync if the alias points at one.
    print(f"bootstrap {boot['pair'][0]}-{boot['pair'][1]}: "
          f"gap={boot['point_gap']}  95% CI {boot['ci95']}  "
          f"P(gap>0)={boot['p_positive']}")
    return boot


def main(base_seed: int = 0) -> None:
    start = empty_state(RES, RES)
    game_value(start)                                # warm the shared table
    oracle = Oracle()

    names = sorted(ROSTER)
    score: Dict[str, float] = defaultdict(float)     # 1 / 0.5 / 0 per game
    wdl_tally = {n: [0, 0, 0] for n in names}        # wins, draws, losses
    n_games: Dict[Tuple[str, str], int] = defaultdict(int)
    dec = {n: [0, 0, 0, 0] for n in names}           # decisions, wdl_regret,
                                                     # critical, crit_wdl_regret

    h2h = {a: {b: [0, 0, 0] for b in names if b != a} for a in names}

    pair_id = 0
    for a in names:
        for b in names:
            if a == b:
                continue
            pair_id += 1
            for g in range(GAMES_PER_ORDERED_PAIR):
                seed = base_seed * 10_000_000 + pair_id * 100_000 + g
                p0 = ROSTER[a](seed)
                p1 = ROSTER[b](seed + 50_000)
                result = play_game(p0, p1, start, oracle=oracle, track_seats=(0, 1))
                u = orient(result.terminal_value, 0)
                sa, sb = (1.0, 0.0) if u > 0 else (0.0, 1.0) if u < 0 else (0.5, 0.5)
                score[a] += sa
                score[b] += sb
                ia = 0 if sa == 1 else (1 if sa == 0.5 else 2)
                ib = 0 if sb == 1 else (1 if sb == 0.5 else 2)
                wdl_tally[a][ia] += 1
                wdl_tally[b][ib] += 1
                h2h[a][b][ia] += 1
                h2h[b][a][ib] += 1
                n_games[(a, b)] += 1
                n_games[(b, a)] += 1
                for d in result.decisions:
                    n = a if d.mover == 0 else b
                    dec[n][0] += 1
                    dec[n][1] += d.wdl_regret
                    if d.is_critical:
                        dec[n][2] += 1
                        dec[n][3] += d.wdl_regret
            print(f"  {a} vs {b}: done")

    elo = fit_elo(dict(score), dict(n_games))

    rows = []
    for n in sorted(names, key=lambda x: -elo[x]):
        w, dr, l = wdl_tally[n]
        rows.append({
            "agent": n,
            "elo": round(elo[n], 1),
            "games": w + dr + l,
            "wins": w, "draws": dr, "losses": l,
            "mean_wdl_regret": round(dec[n][1] / dec[n][0], 4),
            "mean_wdl_regret_critical": round(dec[n][3] / dec[n][2], 4) if dec[n][2] else None,
            "worst_case": WORST_CASE.get(n, "n/a (stochastic policy)"),
        })

    print(f"\n{'agent':<14}{'Elo':>7}{'W/D/L':>16}{'regret':>9}"
          f"{'crit-regret':>13}  worst case")
    for row in rows:
        print(f"{row['agent']:<14}{row['elo']:>7.1f}"
              f"{row['wins']:>6}/{row['draws']}/{row['losses']:<4}"
              f"{row['mean_wdl_regret']:>9.4f}"
              f"{row['mean_wdl_regret_critical']:>13.4f}  {row['worst_case']}")

    h2h_out = {a: {b: {"w": v[0], "d": v[1], "l": v[2]}
                   for b, v in row.items()}
               for a, row in h2h.items()}
    boot = bootstrap_elo_gap(h2h_out)
    boot["point_gap"] = round(elo["kimi-v2"] - elo["optimal"], 1)
    print(f"\nbootstrap kimi-v2 − optimal: gap={boot['point_gap']}  "
          f"95% CI {boot['ci95']}  P(gap>0)={boot['p_positive']}")

    path = write_result(NAME, {
        "reserves": [RES, RES],
        "games_per_ordered_pair": GAMES_PER_ORDERED_PAIR,
        "roster": names,
        "base_seed": base_seed,
        "anchor": {"rating": ANCHOR_RATING, "virtual_draws": ANCHOR_GAMES},
    }, {"table": rows,
        "head_to_head": h2h_out,
        "bootstrap_kimi_v2_minus_optimal": boot})
    announce(NAME, path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bootstrap":
        bootstrap_recorded()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
