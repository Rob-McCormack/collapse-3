"""Exact best-response certification for deterministic policies (Gate C tool).

For ANY deterministic policy P, seat, and size this solves the one-sided game:
at P's turns the tree takes P's chosen move (one branch); at opponent turns the
opponent minimizes P's win/draw/loss outcome. Repo rule throughout (no legal
move -> immediate attrition end). Because the policy side never branches, the
one-sided tree is a sliver of the full game and certifies or refutes a frozen
strategy in seconds.

Reports, per (policy, size, seat):
  * worst-case outcome for P (win / draw / loss) -- exact, not sampled;
  * the number of one-sided states visited;
  * if P can be forced to lose: one shortest exploit line, its ply length,
    P's first positive-regret move on it (graded against the exact value
    memo), and for each opponent move whether it is optimal or a blunder in
    the TRUE game (available where the enumeration memo is affordable).

This is a first-class, reusable instrument: any proposed "basic strategy" can
be certified (worst case >= draw) or refuted (forced loss + exploit line)
without playing a single sampled game.

Run:  python -m experiments.best_response            # kimi-v1/v2, (3,3)-(5,5)
"""

import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from collapse3.agents import Agent, KimiAgentV1, KimiAgentV2
from collapse3.enumeration import solve_all, wdl
from collapse3.game import (
    GameState,
    Move,
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
)
from experiments._provenance import announce, write_result

NAME = "best_response"

POLICIES: Dict[str, Callable[[], Agent]] = {
    "kimi-v1": KimiAgentV1,
    "kimi-v2": KimiAgentV2,
}


def _end_value(state: GameState) -> Optional[int]:
    t = evaluate_terminal(state)
    if t is not None:
        return t
    if not get_legal_moves(state):
        return attrition_value(state.board)
    return None


def solve_best_response(policy: Agent, r0: int, r1: int, policy_seat: int):
    """Exact worst case for ``policy`` at ``policy_seat`` from the empty board.

    Returns ``(worst_wdl, depth_to_worst, n_states, memo)`` where ``worst_wdl``
    is from the policy's seat (+1 win / 0 draw / -1 loss) and, when the policy
    can be forced to lose, ``depth_to_worst`` is the length in plies of the
    SHORTEST forced loss. NOTE: no symmetry folding -- the policy's peg
    priority breaks the grid symmetry, so states memoize raw.
    """
    memo: Dict[GameState, Tuple[int, int]] = {}   # state -> (wdl, plies)

    def solve(state: GameState) -> Tuple[int, int]:
        cached = memo.get(state)
        if cached is not None:
            return cached
        t = _end_value(state)
        if t is not None:
            out = (wdl(t, policy_seat), 0)
            memo[state] = out
            return out
        if state.turn == policy_seat:
            v, d = solve(apply_move(state, policy.choose(state)))
            out = (v, d + 1)
        else:
            best: Optional[Tuple[int, int]] = None
            for m in get_legal_moves(state):
                v, d = solve(apply_move(state, m))
                # Opponent minimizes policy's outcome; among equal outcomes it
                # prefers the SHORTEST line (yields shortest exploits).
                if best is None or v < best[0] or (v == best[0] and d + 1 < best[1]):
                    best = (v, d + 1)
            out = best
        memo[state] = out
        return out

    root = empty_state(r0, r1, 0)
    worst, depth = solve(root)
    return worst, depth, len(memo), memo


def extract_line(policy: Agent, memo, r0: int, r1: int, policy_seat: int) -> List[Tuple[int, Move]]:
    """Walk one shortest worst-case line: policy plays itself, opponent plays
    the (outcome, then depth)-minimizing move. Returns [(mover, move), ...]."""
    line: List[Tuple[int, Move]] = []
    state = empty_state(r0, r1, 0)
    while _end_value(state) is None:
        if state.turn == policy_seat:
            m = policy.choose(state)
        else:
            m = min(get_legal_moves(state),
                    key=lambda mv: (memo[apply_move(state, mv)][0],
                                    memo[apply_move(state, mv)][1]))
        line.append((state.turn, m))
        state = apply_move(state, m)
    return line


def grade_line(line: List[Tuple[int, Move]], value_memo, r0: int, r1: int,
               policy_seat: int) -> Dict[str, object]:
    """Grade every move on the line against the TRUE game's exact values."""
    state = empty_state(r0, r1, 0)
    graded = []
    first_policy_regret_ply = None
    for ply, (mover, m) in enumerate(line, start=1):
        avals = {mv: wdl(value_memo[apply_move(state, mv)], mover)
                 for mv in get_legal_moves(state)}
        regret = max(avals.values()) - avals[m]
        graded.append({
            "ply": ply,
            "mover": "policy" if mover == policy_seat else "opponent",
            "move": str(m),
            "true_game": "optimal" if regret == 0 else f"blunder(wdl-{regret})",
        })
        if mover == policy_seat and regret > 0 and first_policy_regret_ply is None:
            first_policy_regret_ply = ply
        state = apply_move(state, m)
    return {"moves": graded, "first_policy_positive_regret_ply": first_policy_regret_ply}


def certify(policy_name: str, r: int, policy_seat: int,
            value_memo=None) -> Dict[str, object]:
    policy = POLICIES[policy_name]()
    t0 = time.time()
    worst, depth, n_states, memo = solve_best_response(policy, r, r, policy_seat)
    out: Dict[str, object] = {
        "policy": policy_name,
        "reserves": [r, r],
        "policy_seat": policy_seat,
        "worst_case": {1: "win", 0: "draw", -1: "forced loss"}[worst],
        "one_sided_states": n_states,
        "seconds": round(time.time() - t0, 2),
    }
    if worst == -1:
        line = extract_line(policy, memo, r, r, policy_seat)
        out["shortest_exploit_plies"] = depth
        out["exploit_line"] = [f"{'P' if mv[0] == policy_seat else 'O'}:{mv[1]}" for mv in line]
        if value_memo is not None:
            out["exploit_grading"] = grade_line(line, value_memo, r, r, policy_seat)
    return out


def main(sizes=(3, 4, 5)) -> None:
    rows = []
    for r in sizes:
        value_memo = solve_all(empty_state(r, r)) if r <= 4 else None
        if value_memo is None:
            print(f"\n({r},{r}): exploit-line true-game grading skipped "
                  f"(enumeration too heavy); outcomes are still exact")
        for policy_name in POLICIES:
            for seat in (0, 1):
                row = certify(policy_name, r, seat, value_memo)
                rows.append(row)
                extra = ""
                if "shortest_exploit_plies" in row:
                    extra = f", exploit in {row['shortest_exploit_plies']} plies"
                print(f"({r},{r}) {policy_name} seat P{seat}: "
                      f"{row['worst_case'].upper()}"
                      f" ({row['one_sided_states']:,} states, {row['seconds']}s{extra})")

    path = write_result(NAME, {"sizes": list(sizes), "policies": list(POLICIES)},
                        {"rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else (3, 4, 5))
