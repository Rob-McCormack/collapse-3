"""Memory vs. memoryless floor under hide_reserves.

Three otherwise-identical tabular Q-learners (P0), graded by the exact oracle:

  full                 -- sees (board, res, turn, cooldown)
  hide_reserves        -- sees (board, turn, cooldown)
  hide_reserves_memory -- reserve-blind + destroyed-bead counts from observed moves

Run (simple):
  python -m experiments.memory_floor 4
  python -m experiments.memory_floor 4 --opponent random

Run (5,5) multi-seed / multi-budget sweep for Finding 7:
  python -m experiments.memory_floor 5 --sweep
"""

import argparse
import random
import sys
import time
from typing import Any, Dict, List, Tuple

from collapse3.aliasing import OBSERVATIONS, memory_reserves_obs, regret_floor
from collapse3.enumeration import solve_all, wdl
from collapse3.game import apply_move, empty_state, get_legal_moves
from collapse3.learning import (
    MEMORY_OBSERVATIONS,
    MemoryObsFn,
    QAgent,
    QTable,
    advance_to_agent,
    advance_to_agent_with_memory,
    apply_move_tracked,
    optimal_opponent,
    train_q,
    train_q_with_memory,
)
from experiments._provenance import announce, write_result

NAME = "memory_floor"
OPP_EPS = 0.25

# Repo-canonical (5,5) enumeration checks (FINDINGS / aliasing_floor).
CANONICAL_55 = {
    "n_states": 12_714_999,
    "floor_hide_reserves_all": 0.1677,
}

AGENTS = [
    ("full", "standard", "full"),
    ("hide_reserves", "standard", "hide_reserves"),
    ("hide_reserves_memory", "memory", "hide_reserves_memory"),
]


def random_opponent(rng: random.Random):
    return lambda s: rng.choice(get_legal_moves(s))


def make_opponent(memo, mode: str, rng: random.Random):
    if mode == "random":
        return random_opponent(rng)
    if mode == "optimal":
        return optimal_opponent(memo, 0.0, rng)
    return optimal_opponent(memo, OPP_EPS, rng)


def _audit_stats(n, opt, reg_sum, floor_sum, outcomes):
    return {
        "decisions": n,
        "optimal_rate": round(opt / n, 4) if n else None,
        "mean_regret": round(reg_sum / n, 4) if n else None,
        "matched_floor": round(floor_sum / n, 4) if n else None,
        "excess_over_floor": round((reg_sum - floor_sum) / n, 4) if n else None,
        "win_draw_loss": (outcomes[1], outcomes[0], outcomes[-1]),
    }


def audit_standard(agent, memo, per_state_floor, root, opponent, episodes):
    n = opt = 0
    reg_sum = floor_sum = 0.0
    outcomes = {1: 0, 0: 0, -1: 0}
    for _ in range(episodes):
        s, r = advance_to_agent(root, 0, opponent)
        while r is None:
            moves = get_legal_moves(s)
            a = agent.choose(s)
            avals = {m: wdl(memo[apply_move(s, m)], 0) for m in moves}
            reg = max(avals.values()) - avals[a]
            n += 1
            reg_sum += reg
            opt += (reg == 0)
            floor_sum += per_state_floor.get(s, 0.0)
            s, r = advance_to_agent(apply_move(s, a), 0, opponent)
        outcomes[int(r)] += 1
    return _audit_stats(n, opt, reg_sum, floor_sum, outcomes)


def audit_memory(Q: QTable, obs_fn: MemoryObsFn, memo, per_state_floor, root, opponent, episodes):
    n = opt = 0
    reg_sum = floor_sum = 0.0
    outcomes = {1: 0, 0: 0, -1: 0}
    for _ in range(episodes):
        mem = (0, 0)
        s, mem, r = advance_to_agent_with_memory(root, mem, 0, opponent)
        while r is None:
            key = obs_fn(s, mem)
            moves = get_legal_moves(s)
            a = max(moves, key=lambda m: (Q.get((key, m), 0.0), -moves.index(m)))
            avals = {m: wdl(memo[apply_move(s, m)], 0) for m in moves}
            reg = max(avals.values()) - avals[a]
            n += 1
            reg_sum += reg
            opt += (reg == 0)
            floor_sum += per_state_floor.get(s, 0.0)
            mem = apply_move_tracked(s, a, mem, 0)
            s, mem, r = advance_to_agent_with_memory(apply_move(s, a), mem, 0, opponent)
        outcomes[int(r)] += 1
    return _audit_stats(n, opt, reg_sum, floor_sum, outcomes)


def train_and_audit(
    name: str,
    kind: str,
    mode: str,
    root,
    initial: Tuple[int, int],
    memo,
    per_state_floor,
    opp_train,
    opp_audit,
    episodes: int,
    seed: int,
    audit_episodes: int,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    if kind == "standard":
        Q = train_q(root, mode, episodes, opp_train, agent_seat=0, seed=seed)
        agent = QAgent(Q, mode, name=name)
        rep = audit_standard(agent, memo, per_state_floor, root, opp_audit, audit_episodes)
    else:
        obs_fn = MEMORY_OBSERVATIONS[mode](initial)
        Q = train_q_with_memory(root, obs_fn, episodes, opp_train, agent_seat=0, seed=seed)
        rep = audit_memory(Q, obs_fn, memo, per_state_floor, root, opp_audit, audit_episodes)
    rep["q_entries"] = len(Q)
    rep["train_seconds"] = round(time.time() - t0, 1)
    rep["seed"] = seed
    return rep, time.time() - t0


def summarize_runs(runs: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    vals = [r[field] for r in runs]
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def validate_55(memo, floor_all) -> None:
    n = len(memo)
    floor4 = round(floor_all.floor, 4)
    if n != CANONICAL_55["n_states"]:
        raise SystemExit(
            f"ABORT: reachable states at (5,5) = {n:,}, "
            f"expected {CANONICAL_55['n_states']:,} (repo canonical / FINDINGS)."
        )
    if floor4 != CANONICAL_55["floor_hide_reserves_all"]:
        raise SystemExit(
            f"ABORT: hide_reserves floor (all decisions) = {floor_all.floor:.6f} "
            f"({floor4} to 4dp), expected {CANONICAL_55['floor_hide_reserves_all']}."
        )


def run_sweep_55(audit_episodes: int = 2000):
    r = 5
    initial = (r, r)
    root = empty_state(r, r)

    print(f"Solving all ({r},{r}) states...")
    t0 = time.time()
    memo = solve_all(root)
    solve_s = time.time() - t0
    print(f"  {len(memo):,} states, root value {memo[root]}, {solve_s:.0f}s")

    floor_all = regret_floor(memo, OBSERVATIONS["hide_reserves"], p0_only=False)
    floor_p0 = regret_floor(memo, OBSERVATIONS["hide_reserves"], p0_only=True)
    floor_memory = regret_floor(memo, memory_reserves_obs(initial), p0_only=True)
    print(f"  hide_reserves floor (all decisions): {floor_all.floor:.4f}")
    print(f"  hide_reserves floor (P0 only):       {floor_p0.floor:.4f}")
    print(f"  hide_reserves + destroyed floor:   {floor_memory.floor:.4f}")

    validate_55(memo, floor_all)
    print("  Enumeration checks passed.\n")

    per_state = floor_p0.per_state_regret
    seeds = [0, 1, 2]
    budgets = [300_000, 600_000]
    conditions = []

    for opponent_mode in ("eps_optimal",):
        for episodes in budgets:
            print(f"=== opponent={opponent_mode}, episodes={episodes:,} ===")
            cond = {"opponent": opponent_mode, "episodes": episodes, "agents": {}}
            for name, kind, mode in AGENTS:
                runs = []
                for seed in seeds:
                    opp_train = make_opponent(memo, opponent_mode, random.Random(seed + 100))
                    opp_audit = make_opponent(memo, opponent_mode, random.Random(seed + 200))
                    rep, elapsed = train_and_audit(
                        name, kind, mode, root, initial, memo, per_state,
                        opp_train, opp_audit, episodes, seed, audit_episodes,
                    )
                    rep["audit_opponent_seed"] = seed + 200
                    runs.append(rep)
                    print(f"  {name:>22} seed={seed}: regret {rep['mean_regret']}, "
                          f"opt_rate {rep['optimal_rate']}, {elapsed:.0f}s")
                cond["agents"][name] = {
                    "runs": runs,
                    "mean_regret": summarize_runs(runs, "mean_regret"),
                    "optimal_rate": summarize_runs(runs, "optimal_rate"),
                    "matched_floor": summarize_runs(runs, "matched_floor"),
                }
            conditions.append(cond)
            print()

    # Random opponent at 600K only
    episodes = 600_000
    print(f"=== opponent=random, episodes={episodes:,} ===")
    cond = {"opponent": "random", "episodes": episodes, "agents": {}}
    for name, kind, mode in AGENTS:
        runs = []
        for seed in seeds:
            opp_train = make_opponent(memo, "random", random.Random(seed + 100))
            opp_audit = make_opponent(memo, "random", random.Random(seed + 200))
            rep, elapsed = train_and_audit(
                name, kind, mode, root, initial, memo, per_state,
                opp_train, opp_audit, episodes, seed, audit_episodes,
            )
            rep["audit_opponent_seed"] = seed + 200
            runs.append(rep)
            print(f"  {name:>22} seed={seed}: regret {rep['mean_regret']}, "
                  f"opt_rate {rep['optimal_rate']}, {elapsed:.0f}s")
        cond["agents"][name] = {
            "runs": runs,
            "mean_regret": summarize_runs(runs, "mean_regret"),
            "optimal_rate": summarize_runs(runs, "optimal_rate"),
            "matched_floor": summarize_runs(runs, "matched_floor"),
        }
    conditions.append(cond)

    payload = {
        "n_states": len(memo),
        "solve_seconds": round(solve_s, 1),
        "floor_hide_reserves_all": floor_all.floor,
        "floor_hide_reserves_p0": floor_p0.floor,
        "floor_memory_reserves_p0": floor_memory.floor,
        "seeds": seeds,
        "budgets_eps_optimal": budgets,
        "conditions": conditions,
    }

    config = {
        "reserves": list(initial),
        "sweep": True,
        "seeds": seeds,
        "budgets_eps_optimal": budgets,
        "audit_episodes": audit_episodes,
        "opponent_epsilon": OPP_EPS,
    }
    path = write_result(NAME, config, payload, tag="55")
    announce(NAME, path)
    return payload


def main(r=4, train_episodes=150_000, audit_episodes=2000, opponent="eps_optimal", seed=7):
    initial = (r, r)
    root = empty_state(r, r)

    print(f"Solving all ({r},{r}) states...")
    t0 = time.time()
    memo = solve_all(root)
    print(f"  {len(memo):,} states, root value {memo[root]}, {time.time()-t0:.0f}s")

    floor_blind = regret_floor(memo, OBSERVATIONS["hide_reserves"], p0_only=True)
    floor_memory = regret_floor(memo, memory_reserves_obs(initial), p0_only=True)
    print(f"  hide_reserves floor (uniform, P0): {floor_blind.floor:.4f}")
    print(f"  hide_reserves + destroyed floor:   {floor_memory.floor:.4f}\n")

    per_state = floor_blind.per_state_regret
    opp_train = make_opponent(memo, opponent, random.Random(1))
    opp_audit = make_opponent(memo, opponent, random.Random(2))

    rows = {}
    print(f"Training 3 agents ({train_episodes:,} episodes, opponent={opponent}, seed={seed}):")
    for name, kind, mode in AGENTS:
        rep, elapsed = train_and_audit(
            name, kind, mode, root, initial, memo, per_state,
            opp_train, opp_audit, train_episodes, seed, audit_episodes,
        )
        rows[name] = rep
        print(f"  {name:>22}: regret {rep['mean_regret']}, matched_floor {rep['matched_floor']}, "
              f"W/D/L {rep['win_draw_loss']}  ({elapsed:.0f}s, {rep['q_entries']:,} Q-entries)")

    tag = f"{r}{r}" if r in (4, 5) else None
    path = write_result(
        NAME,
        {
            "reserves": list(initial),
            "train_episodes": train_episodes,
            "audit_episodes": audit_episodes,
            "opponent": opponent,
            "opponent_epsilon": OPP_EPS if opponent == "eps_optimal" else None,
            "seed": seed,
        },
        {
            "n_states": len(memo),
            "floor_hide_reserves_p0": floor_blind.floor,
            "floor_memory_reserves_p0": floor_memory.floor,
            "rows": rows,
        },
        tag=tag if opponent == "eps_optimal" else f"{tag}_random" if tag else None,
    )
    announce(NAME, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reserves", nargs="?", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=150_000)
    ap.add_argument("--audit-episodes", type=int, default=2000)
    ap.add_argument("--opponent", choices=["eps_optimal", "optimal", "random"], default="eps_optimal")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep", action="store_true", help="(5,5) multi-seed/budget sweep")
    args = ap.parse_args(sys.argv[1:] or None)

    if args.sweep:
        if args.reserves != 5:
            raise SystemExit("--sweep requires reserves=5")
        run_sweep_55(args.audit_episodes)
    else:
        main(args.reserves, args.episodes, args.audit_episodes, args.opponent, args.seed)
