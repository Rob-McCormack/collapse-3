"""Sandbagging / misere solves: is throwing the game forceable?

Three exact questions about a player who *wants a bad outcome for themselves*,
solved on the shipped engine (no pass; no legal action -> immediate attrition
end). "Grandpa" is the would-be thrower; "grandson" is the opponent.

1. FORCED THROW (binary minimax). Grandpa tries to force the grandson to WIN;
   the grandson plays adversarially to AVOID winning (prefers draw or a
   grandpa-win). Value 1 = grandson wins, 0 = draw or grandpa-win. Grandpa nodes
   take max, grandson nodes take min. The answer is whether grandpa can *force*
   his own defeat.

2. RANDOM GRANDSON (expectimax). The grandson plays uniformly at random; grandpa
   throws optimally. Value = exact P(grandson wins) -- computed with Fraction, so
   it is an exact rational, not a sample.

3. WEAK THROW (binary minimax). Grandpa tries only to guarantee he NEVER WINS
   (outcome in {draw, grandson win}); the grandson plays adversarially to MAKE
   grandpa win (including reverse-Oops: a grandson removal whose gravity cascade
   completes a line for grandpa, which wins for grandpa automatically). Value
   1 = grandpa does not win. Grandpa nodes take max, grandson nodes take min.

"Grandson wins" = a line win OR an attrition (surviving-bead) win for the
grandson's seat. Draws and grandpa-wins both score 0.

Run:  python -m experiments.sandbagging            # (3,3),(4,4),(5,5)
      python -m experiments.sandbagging 3 4        # only these sizes
"""

import sys
import time
from fractions import Fraction
from typing import Dict, Optional, Tuple

from collapse3.game import (
    GameState,
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    orient,
    tt_key,
)
from experiments._provenance import announce, write_result

NAME = "sandbagging"


class Timeout(Exception):
    """Raised when a solve exceeds its wall-clock budget."""


def _terminal_score(state: GameState) -> Optional[int]:
    """P0-perspective score if the game has ended here, else None.

    Ends on a line/attrition terminal OR when the mover has no legal action
    (immediate attrition end -- the shipped no-pass rule).
    """
    t = evaluate_terminal(state)
    if t is not None:
        return t
    if not get_legal_moves(state):
        return attrition_value(state.board)
    return None


def _seat_wins(score: int, seat: int) -> bool:
    """True iff the final ``score`` is a line or attrition win for ``seat``."""
    return orient(score, seat) > 0


class _Deadline:
    def __init__(self, max_seconds: Optional[float]):
        self.end = None if max_seconds is None else time.monotonic() + max_seconds

    def check(self) -> None:
        if self.end is not None and time.monotonic() > self.end:
            raise Timeout


def forced_throw_forceable(r: int, grandson_seat: int, max_seconds: Optional[float] = None) -> bool:
    """Can grandpa force the grandson to win against an unwilling grandson?"""
    grandpa = 1 - grandson_seat
    memo: Dict[Tuple, int] = {}
    dl = _Deadline(max_seconds)

    def solve(state: GameState) -> int:
        score = _terminal_score(state)
        if score is not None:
            return 1 if _seat_wins(score, grandson_seat) else 0
        key = tt_key(state)
        cached = memo.get(key)
        if cached is not None:
            return cached
        dl.check()
        if state.turn == grandpa:                       # grandpa maximizes toward 1
            val = 0
            for m in get_legal_moves(state):
                if solve(apply_move(state, m)) == 1:
                    val = 1
                    break                               # prune: a 1 is exact for max
        else:                                           # grandson minimizes toward 0
            val = 1
            for m in get_legal_moves(state):
                if solve(apply_move(state, m)) == 0:
                    val = 0
                    break                               # prune: a 0 is exact for min
        memo[key] = val
        return val

    return solve(empty_state(r, r, 0)) == 1


def random_grandson_prob(r: int, grandson_seat: int, max_seconds: Optional[float] = None) -> Fraction:
    """Exact P(grandson wins) with grandson uniform-random, grandpa throwing."""
    grandpa = 1 - grandson_seat
    memo: Dict[Tuple, Fraction] = {}
    dl = _Deadline(max_seconds)

    def solve(state: GameState) -> Fraction:
        score = _terminal_score(state)
        if score is not None:
            return Fraction(1) if _seat_wins(score, grandson_seat) else Fraction(0)
        key = tt_key(state)
        cached = memo.get(key)
        if cached is not None:
            return cached
        dl.check()
        moves = get_legal_moves(state)
        vals = [solve(apply_move(state, m)) for m in moves]
        if state.turn == grandpa:                       # grandpa maximizes P
            val = max(vals)
        else:                                           # grandson plays uniformly
            val = sum(vals, Fraction(0)) / len(vals)
        memo[key] = val
        return val

    return solve(empty_state(r, r, 0))


def weak_throw_forceable(r: int, grandpa_seat: int, max_seconds: Optional[float] = None) -> bool:
    """Can grandpa guarantee he never wins, vs a grandson trying to make him win?"""
    grandson = 1 - grandpa_seat
    memo: Dict[Tuple, int] = {}
    dl = _Deadline(max_seconds)

    def solve(state: GameState) -> int:
        score = _terminal_score(state)
        if score is not None:
            return 0 if _seat_wins(score, grandpa_seat) else 1
        key = tt_key(state)
        cached = memo.get(key)
        if cached is not None:
            return cached
        dl.check()
        if state.turn == grandpa_seat:                  # grandpa wants outcome 1 (no win)
            val = 0
            for m in get_legal_moves(state):
                if solve(apply_move(state, m)) == 1:
                    val = 1
                    break
        else:                                           # grandson tries to make grandpa win (0)
            val = 1
            for m in get_legal_moves(state):
                if solve(apply_move(state, m)) == 0:
                    val = 0
                    break
        memo[key] = val
        return val

    return solve(empty_state(r, r, 0)) == 1


def _prob_pair(r: int, cap: Optional[float]) -> Dict[str, object]:
    first = random_grandson_prob(r, grandson_seat=0, max_seconds=cap)
    second = random_grandson_prob(r, grandson_seat=1, max_seconds=cap)
    return {
        "grandson_first": float(first),
        "grandson_second": float(second),
        "grandson_first_exact": str(first),
        "grandson_second_exact": str(second),
    }


def solve_size(r: int, forced_cap: Optional[float] = None, weak_cap: Optional[float] = None,
               prob_cap: Optional[float] = None) -> Dict[str, object]:
    out: Dict[str, object] = {}

    t0 = time.monotonic()
    try:
        out["forced_throw"] = {
            "grandson_first": forced_throw_forceable(r, 0, forced_cap),
            "grandson_second": forced_throw_forceable(r, 1, forced_cap),
        }
        out["forced_throw_seconds"] = round(time.monotonic() - t0, 2)
    except Timeout:
        out["forced_throw"] = None
        out["forced_throw_seconds"] = f">{forced_cap} (timed out)"

    t0 = time.monotonic()
    try:
        out["random_grandson_win_prob"] = _prob_pair(r, prob_cap)
        out["random_grandson_seconds"] = round(time.monotonic() - t0, 2)
    except Timeout:
        out["random_grandson_win_prob"] = None
        out["random_grandson_seconds"] = f">{prob_cap} (timed out)"

    t0 = time.monotonic()
    try:
        out["weak_throw"] = {
            "grandpa_first": weak_throw_forceable(r, 0, weak_cap),
            "grandpa_second": weak_throw_forceable(r, 1, weak_cap),
        }
        out["weak_throw_seconds"] = round(time.monotonic() - t0, 2)
    except Timeout:
        out["weak_throw"] = None
        out["weak_throw_seconds"] = f">{weak_cap} (timed out)"

    return out


def main(sizes=(3, 4, 5)) -> None:
    forced_cap = 45 * 60      # 45 min per seat, per Fable's guard
    weak_cap = 45 * 60
    prob_cap = 30 * 60        # expectimax can't prune; cap the (5,5) rational solve
    by_size: Dict[str, object] = {}
    for r in sizes:
        print(f"\n=== reserves ({r},{r}) ===")
        res = solve_size(r, forced_cap=forced_cap, weak_cap=weak_cap, prob_cap=prob_cap)
        ft = res["forced_throw"]
        pr = res["random_grandson_win_prob"]
        wt = res["weak_throw"]
        print(f"  forced throw     : {ft}  ({res['forced_throw_seconds']}s)")
        if pr is None:
            print(f"  P(grandson wins) : (timed out, {res['random_grandson_seconds']})")
        else:
            print(f"  P(grandson wins) : first={pr['grandson_first']:.4f} "
                  f"second={pr['grandson_second']:.4f}  ({res['random_grandson_seconds']}s)")
        print(f"  weak throw       : {wt}  ({res['weak_throw_seconds']}s)")
        by_size[f"{r}_{r}"] = res

    path = write_result(NAME, {"sizes": list(sizes)}, {"by_size": by_size})
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else (3, 4, 5))
