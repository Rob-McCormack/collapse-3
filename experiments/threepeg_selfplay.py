"""Self-play on the Three-Peg sibling, audited against the exact oracle.

Every learned-agent result elsewhere in the repo trains *against the oracle*
(``optimal_opponent``). This experiment does the opposite: **true self-play** --
one shared, evolving Q-table drives BOTH seats, with zero oracle access during
training (Monte-Carlo, terminal reward only, gamma=1). The oracle is used ONLY
afterwards, to grade the frozen policy exactly.

The Three-Peg sibling (Finding 12) is exactly solvable at every reserve size, so
we can grade a self-play agent across the whole phase boundary -- which the full
board (enumerable only to (5,5)) never permits. Everything is un-folded: the
sibling is not grid-symmetric, so no D4 folding (we reuse ``solve_all`` and the
un-folded ``best_response`` certifier, never the folded Oracle/solver).

Three exact measurements, multi-seed, both seats, across the (6,6)-(7,7) boundary:

  1. SATURATION vs GLOBAL EROSION. Self-play reaches ~0 regret on its OWN
     trajectories while its GLOBAL regret (uniform over all reachable decisions)
     stays higher and its edge over a random-policy baseline ERODES with size
     (it removes ~2/3 of the baseline's regret small, <1/2 large), with coverage
     falling too. The baseline is printed beside the regret so the number has a
     scale and coverage is never blended into "competence" (Finding 5 denominator
     discipline).

  2. EXPLOITABILITY (exact best response). Freezing the self-play policy and
     letting the worst-case adversary reply (``best_response.solve_best_response``)
     certifies whether it can be forced to lose -- despite flat self-play records.

  3. TRANSFER ACROSS THE BOUNDARY. A reserve-blind (``hide_reserves``) self-play
     policy has a size-independent opening (the empty-board observation is
     identical at every reserve). Trained at (5,5) -- where the centre uniquely
     wins -- it carries "play centre" into (7,7), where the centre LOSES to a P1
     line (Finding 12/13). Distribution shift onto the exact enumerated boundary.

SCOPE: sibling = PARALLEL evidence, never a fact about Collapse3. Fixed seeds
make every number here deterministic and reproducible.

Run:  python -m experiments.threepeg_selfplay              # sizes 4-7, 5 seeds
      python -m experiments.threepeg_selfplay 4 5          # a subset of sizes
"""

import random
import sys
import time
from typing import Dict, Tuple

from collapse3.enumeration import solve_all, wdl
from collapse3.game import (
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    placement_pegs,
)
from collapse3.learning import OBSERVATIONS, QAgent, _wdl, advance_to_agent, optimal_opponent
from experiments._provenance import announce, write_result
from experiments.best_response import solve_best_response
from experiments.threepeg_floor import EXPECTED_FLOORS, PLACEMENT_ROW

NAME = "threepeg_selfplay"
QTable = Dict[Tuple, float]

SEEDS = (0, 1, 2, 3, 4)
REGIMES = ("full", "hide_reserves")
_WDL_LABEL = {1: "win", 0: "draw", -1: "loss"}
_PEG_LABEL = {0: "end", 1: "centre", 2: "end"}


def _greedy(Q: QTable, key, moves):
    """Greedy move under Q, deterministic tie-break (engine-order first move)."""
    return max(moves, key=lambda m: (Q.get((key, m), 0.0), -moves.index(m)))


def train_selfplay(root, obs_fn, episodes, seed, eps0=0.4, lr0=0.25):
    """Monte-Carlo self-play. One shared Q-table drives both seats (the obs key
    includes ``turn``, so the seats occupy disjoint keys). Each decision is
    updated toward the terminal value oriented to its own mover. Returns
    ``(Q, visited_state_count)``.
    """
    rng = random.Random(seed)
    Q: QTable = {}
    visited = set()
    for ep in range(episodes):
        frac = ep / episodes if episodes else 1.0
        eps = max(0.05, eps0 * (1 - frac))
        lr = max(0.02, lr0 * (1 - frac))
        s = root
        history = []
        while True:
            visited.add(s)
            t = evaluate_terminal(s)
            if t is not None:
                term = t
                break
            moves = get_legal_moves(s)
            if not moves:
                term = attrition_value(s.board)
                break
            mover = s.turn
            key = obs_fn(s)
            a = rng.choice(moves) if rng.random() < eps else _greedy(Q, key, moves)
            history.append((mover, key, a))
            s = apply_move(s, a)
        for mover, key, a in history:
            g = float(_wdl(term, mover))
            Q[(key, a)] = Q.get((key, a), 0.0) + lr * (g - Q.get((key, a), 0.0))
    return Q, len(visited)


def audit_uniform(Q: QTable, obs_fn, memo) -> float:
    """Mean WDL regret of the greedy policy over ALL reachable decision states."""
    n = 0
    reg = 0.0
    for s in memo:
        if evaluate_terminal(s) is not None:
            continue
        moves = get_legal_moves(s)
        if not moves:
            continue
        mover = s.turn
        avals = {m: wdl(memo[apply_move(s, m)], mover) for m in moves}
        a = _greedy(Q, obs_fn(s), moves)
        reg += max(avals.values()) - avals[a]
        n += 1
    return reg / n if n else 0.0


def random_uniform_baseline(memo) -> float:
    """Exact expected uniform regret of a uniformly-random legal policy over ALL
    reachable decision states -- the scale bar for ``audit_uniform``. Closed-form
    (no sampling): a random mover's expected regret in a state is the mean regret
    of its legal moves. Same denominator as ``audit_uniform`` so they compare.
    """
    n = 0
    reg = 0.0
    for s in memo:
        if evaluate_terminal(s) is not None:
            continue
        moves = get_legal_moves(s)
        if not moves:
            continue
        mover = s.turn
        avals = [wdl(memo[apply_move(s, m)], mover) for m in moves]
        best = max(avals)
        reg += sum(best - v for v in avals) / len(avals)
        n += 1
    return reg / n if n else 0.0


def audit_onpolicy_self(Q: QTable, obs_fn, memo, root, games, seed, eps=0.1) -> float:
    """Mean regret vs the oracle along the agent's OWN eps-noisy self-play games."""
    rng = random.Random(seed)
    n = 0
    reg = 0.0
    for _ in range(games):
        s = root
        while evaluate_terminal(s) is None:
            moves = get_legal_moves(s)
            if not moves:
                break
            mover = s.turn
            avals = {m: wdl(memo[apply_move(s, m)], mover) for m in moves}
            a = rng.choice(moves) if rng.random() < eps else _greedy(Q, obs_fn(s), moves)
            reg += max(avals.values()) - avals[a]
            n += 1
            s = apply_move(s, a)
    return reg / n if n else 0.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def analyse_regime(r, obs_name, memo, episodes, games):
    """Multi-seed self-play at (r,r) under one observation regime."""
    obs_fn = OBSERVATIONS[obs_name]
    root = empty_state(r, r)
    total_states = len(memo)
    per = {"coverage": [], "uniform_regret": [], "onpolicy_self": [],
           "best_response_seat0": [], "best_response_seat1": []}
    for seed in SEEDS:
        Q, visited = train_selfplay(root, obs_fn, episodes, seed=seed)
        per["coverage"].append(round(visited / total_states, 4))
        per["uniform_regret"].append(round(audit_uniform(Q, obs_fn, memo), 4))
        per["onpolicy_self"].append(round(audit_onpolicy_self(Q, obs_fn, memo, root, games, seed), 4))
        agent = QAgent(Q, obs_name)
        for seat in (0, 1):
            worst, _depth, _n, _m = solve_best_response(agent, r, r, seat)
            per[f"best_response_seat{seat}"].append(worst)
    return {
        "coverage_mean": round(_mean(per["coverage"]), 4),
        "uniform_regret_mean": round(_mean(per["uniform_regret"]), 4),
        "uniform_regret_min": min(per["uniform_regret"]),
        "uniform_regret_max": max(per["uniform_regret"]),
        "onpolicy_self_mean": round(_mean(per["onpolicy_self"]), 4),
        "exploitable_seat0": sum(1 for w in per["best_response_seat0"] if w < 0),
        "exploitable_seat1": sum(1 for w in per["best_response_seat1"] if w < 0),
        "not_won_seat0": sum(1 for w in per["best_response_seat0"] if w < 1),
        "per_seed": per,
    }


def transfer_across_boundary(memos, episodes, train_r=5, test_sizes=(5, 6, 7)):
    """A reserve-blind self-play policy trained at (train_r,train_r): read its
    size-independent opening and certify it at larger sizes across the boundary.
    """
    obs_name = "hide_reserves"
    obs_fn = OBSERVATIONS[obs_name]
    root = empty_state(train_r, train_r)
    per_seed = []
    for seed in SEEDS:
        Q, _ = train_selfplay(root, obs_fn, episodes, seed=seed)
        moves = get_legal_moves(empty_state(train_r, train_r))
        open_move = _greedy(Q, obs_fn(empty_state(train_r, train_r)), moves)
        open_peg = open_move[1]
        agent = QAgent(Q, obs_name)
        row = {"seed": seed, "opening_peg": open_peg, "opening": _PEG_LABEL[open_peg]}
        for tr in test_sizes:
            child = apply_move(empty_state(tr, tr), open_move)
            row[f"opening_value_{tr}_{tr}"] = _WDL_LABEL[wdl(memos[tr][child], 0)]
            worst, _d, _n, _m = solve_best_response(agent, tr, tr, 0)
            row[f"best_response_{tr}_{tr}_seat0"] = _WDL_LABEL[worst]
        per_seed.append(row)
    centre = sum(1 for row in per_seed if row["opening_peg"] == 1)
    forced_loss_7 = sum(1 for row in per_seed if row.get("best_response_7_7_seat0") == "loss")
    return {"trained_at": f"{train_r}_{train_r}", "regime": obs_name,
            "test_sizes": list(test_sizes), "per_seed": per_seed,
            "seeds_open_centre": centre, "seeds_forced_loss_at_7_7": forced_loss_7}


def main(sizes=(4, 5, 6, 7), episodes=50_000, games=300):
    with placement_pegs(PLACEMENT_ROW):
        print(f"Three-Peg self-play (placements {list(PLACEMENT_ROW)}), "
              f"{episodes:,} MC episodes/seed, seeds={list(SEEDS)}\n")
        memos = {r: solve_all(empty_state(r, r)) for r in sorted(set(sizes) | {5, 6, 7})}

        by_size = {}
        # P1 is theoretically LOST at every size (the empty-board root is a P0
        # forced win from (3,3) up), so seat-1 best-response is always a loss --
        # the game value, not an exploit. The informative exploitability signal
        # is at the winnable seat 0: "not_won" = forced below a win (to draw/loss).
        print(f"{'size':>6} {'regime':>14} {'cover':>7} {'uniform(min..max)':>22} "
              f"{'random':>7} {'onpol':>7} {'s0 not-won':>10}")
        for r in sizes:
            memo = memos[r]
            floor = EXPECTED_FLOORS.get(r, (None, None, None, None))
            baseline = round(random_uniform_baseline(memo), 4)
            regimes = {}
            for obs_name in REGIMES:
                t0 = time.time()
                rep = analyse_regime(r, obs_name, memo, episodes, games)
                regimes[obs_name] = rep
                print(f"({r:>2},{r:<2}) {obs_name:>14} {rep['coverage_mean']:>7.3f} "
                      f"{rep['uniform_regret_mean']:>8.4f}"
                      f" ({rep['uniform_regret_min']:.3f}..{rep['uniform_regret_max']:.3f}) "
                      f"{baseline:>7.4f} "
                      f"{rep['onpolicy_self_mean']:>7.4f} "
                      f"{rep['not_won_seat0']:>3}/{len(SEEDS)}     [{time.time()-t0:.0f}s]")
            by_size[f"{r}_{r}"] = {"states": len(memo),
                                   "exact_floor_hide_reserves": floor[3],
                                   "random_uniform_regret": baseline,
                                   "regimes": regimes}

        print("\nTransfer across the boundary (reserve-blind, trained at (5,5)):")
        transfer = transfer_across_boundary(memos, episodes)
        for row in transfer["per_seed"]:
            print(f"  seed {row['seed']}: opens {row['opening']:>6} (peg {row['opening_peg']}) | "
                  f"(5,5) {row['opening_value_5_5']:>4} -> (6,6) {row['opening_value_6_6']:>4} "
                  f"-> (7,7) {row['opening_value_7_7']:>4} | "
                  f"best-response (7,7) seat0: {row['best_response_7_7_seat0'].upper()}")
        print(f"  => {transfer['seeds_open_centre']}/{len(SEEDS)} seeds learn the centre opening; "
              f"{transfer['seeds_forced_loss_at_7_7']}/{len(SEEDS)} are forced to lose at (7,7).")

    payload = {"variant": "Three-Peg Collapse (self-play)",
               "placement_pegs": list(PLACEMENT_ROW), "seeds": list(SEEDS),
               "episodes": episodes, "audit_games": games,
               "by_size": by_size, "transfer": transfer}
    path = write_result(NAME, {"sizes": list(sizes), "episodes": episodes, "seeds": list(SEEDS)}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else (4, 5, 6, 7))
