"""Frozen plan vs. re-solving, under enumerated opponent blunders.

The root of (r, r) is a draw under perfect play. Player A (P0) either REPLAYS a
frozen principal-line plan (falling back to the first legal move if the planned
move is illegal in the actual position) or RE-SOLVES every position from
scratch. Player B (P1) plays optimally except for one deliberate deviation: at
its t-th decision it plays its k-th-best non-optimal move, then returns to
optimal play.

We enumerate every (t, k) deviation. Any B win against the *static* A is an
inversion: a deliberately worse move beating an exact-oracle-derived plan out of
a drawn game -- because the frozen plan was a best-response to the principal
line, not a strategy robust to off-line play.

Run:  python -m experiments.frozen_plan_vs_blunder        # default r=5
      python -m experiments.frozen_plan_vs_blunder 4
"""

import sys
import time

from collapse3.game import (
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
)
from collapse3.solver import game_value, reset_search_state
from experiments._provenance import announce, write_result

NAME = "frozen_plan_vs_blunder"


def exact(state):
    t = evaluate_terminal(state)
    return t if t is not None else game_value(state)


def ordered_by_value(state):
    """Legal moves, best-first for the mover (deterministic tie-break by index)."""
    p = state.turn
    scored = [(exact(apply_move(state, m)), i, m) for i, m in enumerate(get_legal_moves(state))]
    scored.sort(key=lambda x: (-x[0] if p == 0 else x[0], x[1]))
    return [m for _, _, m in scored]


def principal_a_moves(r):
    s = empty_state(r, r)
    plan = []
    while evaluate_terminal(s) is None:
        moves = get_legal_moves(s)
        if not moves:
            break
        m = ordered_by_value(s)[0]
        if s.turn == 0:
            plan.append(m)
        s = apply_move(s, m)
    return plan


def play(r, a_mode, plan, dev_t, dev_k):
    s = empty_state(r, r)
    a_i = b_i = 0
    applied = False
    while True:
        t = evaluate_terminal(s)
        if t is not None:
            return t, applied
        moves = get_legal_moves(s)
        if not moves:                      # immediate end -> attrition
            return attrition_value(s.board), applied
        if s.turn == 0:
            if a_mode == "resolve":
                m = ordered_by_value(s)[0]
            else:
                m = plan[a_i] if a_i < len(plan) and plan[a_i] in moves else moves[0]
                a_i += 1
        else:
            ranked = ordered_by_value(s)
            if b_i == dev_t and len(ranked) > 1 and dev_k < len(ranked) - 1:
                m = ranked[1 + dev_k]
                applied = True
            else:
                m = ranked[0]
            b_i += 1
        s = apply_move(s, m)


def main(r=5, max_b_turns=12, max_alts=16):
    reset_search_state()
    game_value(empty_state(r, r))  # warm TT

    plan = principal_a_moves(r)
    base_static, _ = play(r, "static", plan, -1, -1)
    base_resolve, _ = play(r, "resolve", plan, -1, -1)

    rows = []
    for t in range(max_b_turns):
        for k in range(max_alts):
            v_s, applied = play(r, "static", plan, t, k)
            if not applied:
                continue
            v_r, _ = play(r, "resolve", plan, t, k)
            rows.append((t, k, v_s, v_r))

    def counts(idx):
        a = sum(1 for row in rows if row[idx] > 0)   # A wins
        d = sum(1 for row in rows if row[idx] == 0)
        b = sum(1 for row in rows if row[idx] < 0)   # B wins
        return a, d, b

    a_s, d_s, w_s = counts(2)
    a_r, d_r, w_r = counts(3)
    inversions = [(t, k) for (t, k, vs, vr) in rows if vs < 0]

    print(f"Frozen plan vs re-solving under blunders, reserves ({r},{r})")
    print("=" * 68)
    print(f"  root value: {base_static} (draw={base_static == 0})")
    print(f"  enumerated deviations: {len(rows)}")
    print(f"  vs STATIC A    : A {a_s}, draw {d_s}, B WINS {w_s}")
    print(f"  vs RE-SOLVING A: A {a_r}, draw {d_r}, B WINS {w_r}")
    print(f"  inversions (worse B move beats frozen plan): {len(inversions)}")
    if inversions:
        print(f"    at (b_turn, alt): {inversions[:12]}{' ...' if len(inversions) > 12 else ''}")

    payload = {
        "reserves": [r, r],
        "root_value": base_static,
        "deviations": len(rows),
        "static_A": {"A_wins": a_s, "draws": d_s, "B_wins": w_s},
        "resolving_A": {"A_wins": a_r, "draws": d_r, "B_wins": w_r},
        "inversions": inversions,
    }
    path = write_result(NAME, {"reserves": [r, r], "max_b_turns": max_b_turns, "max_alts": max_alts}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
