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

import sys
from collections import defaultdict
from typing import Dict, Tuple

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

    path = write_result(NAME, {
        "reserves": [RES, RES],
        "games_per_ordered_pair": GAMES_PER_ORDERED_PAIR,
        "roster": names,
        "base_seed": base_seed,
        "anchor": {"rating": ANCHOR_RATING, "virtual_draws": ANCHOR_GAMES},
    }, {"table": rows,
        "head_to_head": {a: {b: {"w": v[0], "d": v[1], "l": v[2]}
                             for b, v in row.items()}
                         for a, row in h2h.items()}})
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
