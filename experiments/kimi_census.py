"""Distribution-free census of KimiAgent v1/v2 over ALL reachable decision
states (uniform measure) at (3,3) and (4,4).

For every reachable non-terminal state with a legal move, grade the policy's
chosen move against the exact enumeration: optimal or not, WDL regret, split
by criticality, game phase (beads on board), and chosen move kind. The
purpose is to locate WHERE each frozen policy is wrong -- in particular
whether v2's (4,4) errors concentrate in post-land-grab midgame states, the
mechanism behind the strategy-complexity crossover (Finding 8).

Run:  python -m experiments.kimi_census
"""

from collections import defaultdict
from typing import Dict

from collapse3.agents import KimiAgentV1, KimiAgentV2
from collapse3.enumeration import reachable_states, solve_all, wdl
from collapse3.game import apply_move, empty_state, evaluate_terminal, get_legal_moves
from experiments._provenance import announce, write_result

NAME = "kimi_census"


def _phase(state) -> str:
    beads = sum(len(p) for p in state.board)
    if beads <= 2:
        return "opening(0-2)"
    if beads <= 4:
        return "land-grab(3-4)"
    if beads <= 6:
        return "midgame(5-6)"
    return "late(7+)"


def census(policy, r: int, memo) -> Dict[str, object]:
    n = n_opt = n_crit = n_crit_opt = 0
    total_wdl = total_wdl_crit = 0
    by_phase = defaultdict(lambda: [0, 0, 0])       # n, optimal, wdl regret
    by_kind = defaultdict(lambda: [0, 0])           # n, optimal
    worst = []                                       # regret-2 examples

    for state in reachable_states(empty_state(r, r)):
        if evaluate_terminal(state) is not None:
            continue
        moves = get_legal_moves(state)
        if not moves:
            continue
        mover = state.turn
        avals = {m: wdl(memo[apply_move(state, m)], mover) for m in moves}
        best = max(avals.values())
        critical = min(avals.values()) < best

        chosen = policy.choose(state)
        regret = best - avals[chosen]
        opt = regret == 0

        n += 1
        n_opt += opt
        total_wdl += regret
        ph = _phase(state)
        by_phase[ph][0] += 1
        by_phase[ph][1] += opt
        by_phase[ph][2] += regret
        by_kind[chosen[0]][0] += 1
        by_kind[chosen[0]][1] += opt
        if critical:
            n_crit += 1
            n_crit_opt += opt
            total_wdl_crit += regret
        if regret == 2 and len(worst) < 5:
            worst.append({"board": [list(p) for p in state.board],
                          "turn": state.turn, "res": list(state.res),
                          "cooldown": list(state.cooldown), "chosen": str(chosen)})

    return {
        "decision_states": n,
        "optimal_rate": round(n_opt / n, 4),
        "mean_wdl_regret": round(total_wdl / n, 4),
        "critical_states": n_crit,
        "optimal_rate_critical": round(n_crit_opt / n_crit, 4) if n_crit else None,
        "mean_wdl_regret_critical": round(total_wdl_crit / n_crit, 4) if n_crit else None,
        "by_phase": {k: {"n": v[0], "opt_rate": round(v[1] / v[0], 4),
                         "mean_wdl_regret": round(v[2] / v[0], 4)}
                     for k, v in sorted(by_phase.items())},
        "by_chosen_kind": {k: {"n": v[0], "opt_rate": round(v[1] / v[0], 4)}
                           for k, v in sorted(by_kind.items())},
        "worst_examples": worst,
    }


def main(sizes=(3, 4)) -> None:
    out = {}
    for r in sizes:
        memo = solve_all(empty_state(r, r))
        for name, cls in (("kimi-v1", KimiAgentV1), ("kimi-v2", KimiAgentV2)):
            row = census(cls(), r, memo)
            out[f"{name}@({r},{r})"] = row
            print(f"({r},{r}) {name}: {row['decision_states']:,} decision states, "
                  f"opt {row['optimal_rate']}, crit-opt {row['optimal_rate_critical']}, "
                  f"crit-wdl {row['mean_wdl_regret_critical']}")
            for ph, v in row["by_phase"].items():
                print(f"    {ph:<14} n={v['n']:>8,}  opt={v['opt_rate']:.4f}  "
                      f"wdl={v['mean_wdl_regret']:.4f}")

    path = write_result(NAME, {"sizes": list(sizes)}, out)
    announce(NAME, path)


if __name__ == "__main__":
    main()
