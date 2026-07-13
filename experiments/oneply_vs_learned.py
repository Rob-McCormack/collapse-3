"""One-ply heuristic vs. machine-learned agent, both graded by the oracle.

Two very different weak agents, audited identically at reserves (r, r):

  * one-ply heuristic -- sees exactly one move ahead (immediate win/loss + a
    static line-threat/material heuristic). No search, no learning.
  * learned (tabular Q) -- trained from terminal reward only vs an eps-optimal
    opponent, zero oracle access.

Both are scored by per-decision value-based (win/draw/loss) regret against the
exact value map while playing the eps-optimal opponent, then played head-to-head.
Illustrates that similar win rates can hide very different competence profiles.

Run:  python -m experiments.oneply_vs_learned        # default r=4
"""

import random
import sys
import time

from collapse3.agents import HeuristicOnePlyAgent, OptimalAgent
from collapse3.enumeration import solve_all, wdl
from collapse3.game import apply_move, empty_state, get_legal_moves
from collapse3.learning import QAgent, advance_to_agent, optimal_opponent, train_q
from collapse3.metrics import play_game
from experiments._provenance import announce, write_result

NAME = "oneply_vs_learned"
OPP_EPS = 0.25


def audit(agent, memo, opponent, r, episodes, seed, agent_seat=0):
    rng = random.Random(seed)
    root = empty_state(r, r)
    n = opt = 0
    reg_sum = 0.0
    outcomes = {1: 0, 0: 0, -1: 0}
    for _ in range(episodes):
        s, reward = advance_to_agent(root, agent_seat, opponent)
        while reward is None:
            moves = get_legal_moves(s)
            a = agent.choose(s)
            avals = {m: wdl(memo[apply_move(s, m)], agent_seat) for m in moves}
            reg = max(avals.values()) - avals[a]
            n += 1
            reg_sum += reg
            opt += (reg == 0)
            s, reward = advance_to_agent(apply_move(s, a), agent_seat, opponent)
        outcomes[int(reward)] += 1
    return {
        "optimal_rate": round(opt / n, 4) if n else None,
        "mean_regret": round(reg_sum / n, 4) if n else None,
        "win_draw_loss_vs_eps_optimal": (outcomes[1], outcomes[0], outcomes[-1]),
    }


def main(r=4, train_episodes=150_000, audit_episodes=2000, h2h_games=200):
    print(f"Solving all ({r},{r}) states...")
    t0 = time.time()
    root = empty_state(r, r)
    memo = solve_all(root)
    print(f"  {len(memo):,} states, {time.time()-t0:.0f}s")

    # Train one policy per seat. A Q-table keyed on (obs incl. turn) trained at
    # seat 0 contains NO seat-1 decisions -- playing it as P1 exercises the
    # first-legal-move fallback, not a trained agent (a bug caught in external
    # review). Seat-correct tables are required for the head-to-head.
    print(f"Training Q-learner, seat 0 ({train_episodes:,} episodes)...")
    t0 = time.time()
    Q0 = train_q(root, "full", train_episodes, optimal_opponent(memo, OPP_EPS, random.Random(1)),
                 agent_seat=0, seed=7)
    print(f"  done {time.time()-t0:.0f}s, {len(Q0):,} Q entries")
    print(f"Training Q-learner, seat 1 ({train_episodes:,} episodes)...")
    t0 = time.time()
    Q1 = train_q(root, "full", train_episodes, optimal_opponent(memo, OPP_EPS, random.Random(4)),
                 agent_seat=1, seed=8)
    print(f"  done {time.time()-t0:.0f}s, {len(Q1):,} Q entries\n")

    agents = {
        "one_ply_heuristic": HeuristicOnePlyAgent(seed=0),
        "learned_tabular_q": QAgent(Q0, "full"),
    }
    reports = {}
    print("AUDIT vs exact oracle (playing eps-optimal opponent, agent at seat 0):")
    for name, agent in agents.items():
        rep = audit(agent, memo, optimal_opponent(memo, OPP_EPS, random.Random(2)), r, audit_episodes, seed=3)
        reports[name] = rep
        print(f"  {name:>18}: {rep}")
    rep1 = audit(QAgent(Q1, "full"), memo, optimal_opponent(memo, OPP_EPS, random.Random(5)),
                 r, audit_episodes, seed=6, agent_seat=1)
    reports["learned_tabular_q_seat1"] = rep1
    print(f"  {'learned_q (seat 1)':>18}: {rep1}")

    print(f"\nHEAD-TO-HEAD ({h2h_games} games each color, seat-matched Q-tables):")
    start = empty_state(r, r)
    make_oneply = lambda g: HeuristicOnePlyAgent(seed=g)
    make_learned_p0 = lambda g: QAgent(Q0, "full")
    make_learned_p1 = lambda g: QAgent(Q1, "full")
    h2h = {}
    for label, make_p0, make_p1 in (("oneply_P0_vs_learned_P1", make_oneply, make_learned_p1),
                                    ("learned_P0_vs_oneply_P1", make_learned_p0, make_oneply)):
        w = d = l = 0
        for g in range(h2h_games):
            res = play_game(make_p0(g), make_p1(g), start)
            v = res.terminal_value  # P0 perspective
            w += v > 0
            d += v == 0
            l += v < 0
        h2h[label] = {"P0_wins": w, "draws": d, "P0_losses": l}
        print(f"  {label}: P0 {w}W {d}D {l}L")

    payload = {"reserves": [r, r], "q_entries_seat0": len(Q0), "q_entries_seat1": len(Q1),
               "audit": reports, "head_to_head": h2h}
    path = write_result(NAME, {"reserves": [r, r], "train_episodes": train_episodes}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
