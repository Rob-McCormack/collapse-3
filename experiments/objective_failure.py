"""Objective failure vs. representation failure.

Two ways a learned agent can fall short of optimal play, cleanly separated:

  * Objective failure  -- the *training signal* is too weak. Trained against a
    feeble opponent, terminal reward gives almost no gradient toward optimal
    play, so the agent wins a lot yet carries real regret. This is **fixable**:
    train against a punishing (optimal) opponent and the regret collapses.
  * Representation failure -- the agent can't *see* what governs outcomes. This
    is NOT fixable by a better opponent or more training (see
    experiments/representation_amplification.py and the aliasing floors).

This experiment isolates the first. The SAME full-state tabular learner is
trained twice -- against a random opponent and against an (eps-)optimal one --
then oracle-graded. Representation is held sufficient throughout, so any
difference is attributable to the objective (opponent strength), not the input.

Run:  python -m experiments.objective_failure           # default r=4
"""

import random
import sys
import time

from collapse3.enumeration import solve_all, wdl
from collapse3.game import apply_move, empty_state, get_legal_moves
from collapse3.learning import QAgent, advance_to_agent, optimal_opponent, train_q
from experiments._provenance import announce, write_result

NAME = "objective_failure"
OPP_EPS = 0.25


def random_opponent(rng):
    return lambda s: rng.choice(get_legal_moves(s))


def audit(agent, memo, opponent, r, episodes, seed):
    rng = random.Random(seed)
    root = empty_state(r, r)
    n = opt = 0
    reg_sum = 0.0
    outcomes = {1: 0, 0: 0, -1: 0}
    for _ in range(episodes):
        s, reward = advance_to_agent(root, 0, opponent)
        while reward is None:
            moves = get_legal_moves(s)
            a = agent.choose(s)
            avals = {m: wdl(memo[apply_move(s, m)], 0) for m in moves}
            reg = max(avals.values()) - avals[a]
            n += 1
            reg_sum += reg
            opt += (reg == 0)
            s, reward = advance_to_agent(apply_move(s, a), 0, opponent)
        outcomes[int(reward)] += 1
    return {
        "optimal_rate": round(opt / n, 4) if n else None,
        "mean_regret": round(reg_sum / n, 4) if n else None,
        "win_draw_loss": (outcomes[1], outcomes[0], outcomes[-1]),
    }


def main(r=4, train_episodes=150_000, audit_episodes=2000):
    print(f"Solving all ({r},{r}) states...")
    t0 = time.time()
    root = empty_state(r, r)
    memo = solve_all(root)
    print(f"  {len(memo):,} states, {time.time()-t0:.0f}s\n")

    training_opponents = {
        "trained_vs_random": lambda: random_opponent(random.Random(1)),
        "trained_vs_optimal": lambda: optimal_opponent(memo, OPP_EPS, random.Random(1)),
    }

    rows = {}
    print("Full-state learner (representation is sufficient); only the training "
          "opponent differs:")
    for name, make_train_opp in training_opponents.items():
        t0 = time.time()
        Q = train_q(root, "full", train_episodes, make_train_opp(), seed=7)
        agent = QAgent(Q, "full")
        # "Looks capable": win rate vs a weak (random) opponent.
        looks = audit(agent, memo, random_opponent(random.Random(2)), r, audit_episodes, seed=3)
        # Competence: oracle regret on-trajectory vs a punishing (eps-optimal) opponent.
        comp = audit(agent, memo, optimal_opponent(memo, OPP_EPS, random.Random(2)), r, audit_episodes, seed=3)
        rows[name] = {
            "q_entries": len(Q),
            "win_rate_vs_random": round((looks["win_draw_loss"][0]) / audit_episodes, 3),
            "wdl_vs_random": looks["win_draw_loss"],
            "mean_regret_vs_optimal": comp["mean_regret"],
            "optimal_rate_vs_optimal": comp["optimal_rate"],
            "wdl_vs_optimal": comp["win_draw_loss"],
            # Off-distribution probe: oracle regret on the trajectories a *random*
            # opponent induces -- states the agent rarely saw while training.
            "mean_regret_offdist_random": looks["mean_regret"],
        }
        print(f"  [{name}] Q={len(Q):,} ({time.time()-t0:.0f}s): "
              f"win vs random {rows[name]['win_rate_vs_random']}, "
              f"regret vs optimal {comp['mean_regret']}, "
              f"regret off-dist (vs random) {looks['mean_regret']}, "
              f"W/D/L vs optimal {comp['win_draw_loss']}")

    print("\n  Same representation, same algorithm: training against a punishing "
          "opponent\n  collapses regret. The objective failure is fixable.")
    print("  Off-distribution probe: the vs-optimal agent's regret rises on random-"
          "opponent\n  trajectories -- competence is measured off its training "
          "distribution, not just on it.")

    path = write_result(NAME, {"reserves": [r, r], "train_episodes": train_episodes}, {"rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
