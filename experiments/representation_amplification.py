"""Does optimizing around a lossy representation *amplify* regret?

Trains the SAME tabular Q-learner twice -- full observation vs cooldown-blind --
with terminal reward only and zero oracle access, against an eps-optimal
opponent. The oracle grades afterward. The learned agent's mean regret is
compared to its own structural floor *evaluated over the states it actually
visits* (an apples-to-apples matched floor).

    full obs       -> regret should approach 0 (representation is sufficient)
    cooldown-blind -> regret >= its floor; the question is whether it lands NEAR
                      the floor or well ABOVE it (amplification).

'curve' sweeps the training budget (1x/2x/4x/8x) for the cooldown-blind agent:
if the gap to the floor never closes, the failure is in the input, not the
optimizer.

Run:  python -m experiments.representation_amplification            # amplified
      python -m experiments.representation_amplification curve
      python -m experiments.representation_amplification all --episodes 60000
"""

import argparse
import random
import time

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all, wdl
from collapse3.game import apply_move, empty_state, get_legal_moves
from collapse3.learning import QAgent, advance_to_agent, optimal_opponent, train_q
from experiments._provenance import announce, write_result

NAME = "representation_amplification"
R = 4
OPP_EPS = 0.25


def audit(agent, memo, per_state_floor, opponent, episodes, seed):
    rng = random.Random(seed)
    root = empty_state(R, R)
    n = opt = 0
    reg_sum = floor_sum = 0.0
    outcomes = {1: 0, 0: 0, -1: 0}
    buckets = {}
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
            beads = sum(len(p) for p in s.board)
            phase = "early" if beads <= 2 else ("mid" if beads <= 5 else "late")
            b = buckets.setdefault((phase, a[0]), [0.0, 0])
            b[0] += reg
            b[1] += 1
            s, r = advance_to_agent(apply_move(s, a), 0, opponent)
        outcomes[int(r)] += 1
    return {
        "decisions": n,
        "optimal_rate": round(opt / n, 4) if n else None,
        "mean_regret": round(reg_sum / n, 4) if n else None,
        "matched_floor": round(floor_sum / n, 4) if n else None,
        "excess_over_floor": round((reg_sum - floor_sum) / n, 4) if n else None,
        "win_draw_loss": (outcomes[1], outcomes[0], outcomes[-1]),
        "buckets": {f"{k[0]}/{k[1]}": (round(v[0] / v[1], 4), v[1]) for k, v in sorted(buckets.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", nargs="?", default="amplified", choices=["amplified", "curve", "all"])
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--audit-episodes", type=int, default=2000)
    args = ap.parse_args()

    print(f"Solving all ({R},{R}) states...")
    t0 = time.time()
    root = empty_state(R, R)
    memo = solve_all(root)
    print(f"  {len(memo):,} states, {time.time()-t0:.0f}s")

    floor_cd = regret_floor(memo, OBSERVATIONS["hide_cooldown"], p0_only=True)
    print(f"  cooldown-blind floor (P0): {floor_cd.floor:.4f}\n")

    def make_opp(seed):
        return optimal_opponent(memo, OPP_EPS, random.Random(seed))

    payload = {"reserves": [R, R], "opponent_epsilon": OPP_EPS,
               "floor_hide_cooldown_p0": floor_cd.floor, "amplified": {}, "curve": []}

    if args.test in ("amplified", "all"):
        print(f"AMPLIFIED ({args.episodes:,} episodes):")
        for mode in ("full", "hide_cooldown"):
            t0 = time.time()
            Q = train_q(root, mode, args.episodes, make_opp(1), agent_seat=0, seed=7)
            agent = QAgent(Q, mode)
            floor = regret_floor(memo, OBSERVATIONS[mode], p0_only=True) if mode != "full" else floor_cd
            per_state = floor.per_state_regret if mode != "full" else {}
            rep = audit(agent, memo, per_state, make_opp(2), args.audit_episodes, seed=3)
            payload["amplified"][mode] = {"q_entries": len(Q), **rep}
            print(f"  [{mode}] Q={len(Q):,} trained {time.time()-t0:.0f}s -> "
                  f"regret {rep['mean_regret']}, matched_floor {rep['matched_floor']}, "
                  f"excess {rep['excess_over_floor']}, W/D/L {rep['win_draw_loss']}")
        print()

    if args.test in ("curve", "all"):
        print("CURVE (cooldown-blind budget sweep):")
        per_state = regret_floor(memo, OBSERVATIONS["hide_cooldown"], p0_only=True).per_state_regret
        for mult in (1, 2, 4, 8):
            t0 = time.time()
            eps = args.episodes * mult
            Q = train_q(root, "hide_cooldown", eps, make_opp(1), agent_seat=0, seed=7)
            rep = audit(QAgent(Q, "hide_cooldown"), memo, per_state, make_opp(2), args.audit_episodes, seed=3)
            payload["curve"].append({"multiplier": mult, "episodes": eps, **rep})
            print(f"  {mult}x ({eps:,}): regret {rep['mean_regret']} | "
                  f"floor {rep['matched_floor']} | excess {rep['excess_over_floor']} | "
                  f"W/D/L {rep['win_draw_loss']} | {time.time()-t0:.0f}s")

    path = write_result(NAME, vars(args), payload)
    announce(NAME, path)


if __name__ == "__main__":
    main()
