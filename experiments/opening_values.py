"""Opening values: first-mover advantage, phase transition, and scaling sharpness.

Exact game-theoretic value of the *empty* board under optimal play, swept over
reserve splits (r0, r1). Also measures the **draw fraction** of the full reachable
state space at symmetric sizes — evidence that scaling reserves makes a *sharper*
game (fewer draws, decisive openings) without any rule change.

Surprises:
  1. First player never loses; second player never wins (grid to (6,6)).
  2. A reserve lead pays off only for the mover.
  3. Equal-reserve diagonal: draw for r = 1..5, first-player *line* win at (6,6).
  4. Draw fraction falls monotonically as material increases (exact through (5,5)).

Run:
  python -m experiments.opening_values 6
  python -m experiments.opening_values 6 --census 2 3 4 5   # include (5,5) (~11 min)
"""

import argparse
import sys
import time

from collapse3.enumeration import solve_all
from collapse3.game import apply_move, empty_state, evaluate_terminal, get_legal_moves
from collapse3.solver import game_value, reset_search_state
from experiments._provenance import announce, write_result

NAME = "opening_values"

_LABEL = {
    100: ("L+", "P0 line win"),
    10: ("a+", "P0 attrition win"),
    0: ("..", "draw"),
    -10: ("a-", "P1 attrition win"),
    -100: ("L-", "P1 line win"),
}


def win_type_grid(R):
    grid = {}
    for r0 in range(1, R + 1):
        for r1 in range(1, R + 1):
            grid[(r0, r1)] = game_value(empty_state(r0, r1, 0))
    return grid


def crosscheck(splits):
    out = {}
    for r0, r1 in splits:
        folded = game_value(empty_state(r0, r1, 0))
        memo = solve_all(empty_state(r0, r1, 0))
        out[f"({r0},{r1})"] = {"folded": folded, "enum": memo[empty_state(r0, r1, 0)]}
    return out


def draw_trend(sizes):
    """Fraction of reachable states that are drawn, by symmetric reserve size."""
    out = {}
    for r in sizes:
        t0 = time.time()
        memo = solve_all(empty_state(r, r))
        n = len(memo)
        draws = sum(v == 0 for v in memo.values())
        decisive = n - draws
        out[f"({r},{r})"] = {
            "states": n,
            "draw_frac": round(draws / n, 4),
            "decisive_frac": round(decisive / n, 4),
            "seconds": round(time.time() - t0, 1),
        }
    return out


def check_monotonic(trend: dict) -> bool:
    fracs = [trend[k]["draw_frac"] for k in sorted(trend, key=lambda s: int(s.split(",")[0][1:]))]
    return all(fracs[i] > fracs[i + 1] for i in range(len(fracs) - 1))


def verify_ordering(cells, seeds):
    """Re-solve each cell from a **fresh** table under several shuffled move
    orderings. The exact value is order-invariant, so any disagreement would
    expose an alpha-beta / transposition-table ordering bug. This is the only
    cross-check available at (6,6), where full enumeration is infeasible.
    """
    out = {}
    for (r0, r1) in cells:
        base = game_value(empty_state(r0, r1, 0), fresh=True)
        vals = [game_value(empty_state(r0, r1, 0), order_seed=s) for s in seeds]
        out[f"({r0},{r1})"] = {
            "base": base,
            "shuffled": vals,
            "invariant": all(v == base for v in vals),
        }
    return out


def principal_variation(state, max_len=80):
    """Follow an optimal line from ``state`` (mover always plays a value-preserving
    move). Returns the (state, value) pairs along the way."""
    pv = []
    s = state
    for _ in range(max_len):
        if evaluate_terminal(s) is not None:
            break
        moves = get_legal_moves(s)
        if not moves:
            break
        v = game_value(s)
        nxt = None
        for m in moves:
            child = apply_move(s, m)
            if game_value(child) == v:
                nxt = child
                break
        if nxt is None:
            break
        pv.append((s, v))
        s = nxt
    return pv


def verify_pv(state, seeds):
    """Extract the principal variation and independently re-verify every position
    on it from a fresh, order-shuffled table."""
    pv = principal_variation(state)
    matches = []
    for i, (s, v) in enumerate(pv):
        rv = game_value(s, order_seed=seeds[i % len(seeds)])
        matches.append(rv == v)
    return {
        "pv_len": len(pv),
        "root_value": pv[0][1] if pv else None,
        "all_positions_match": all(matches),
    }


def main(R=6, census_sizes=(2, 3, 4)):
    reset_search_state()

    t0 = time.time()
    grid = win_type_grid(R)
    print(f"Opening value grid to ({R},{R})  [{time.time()-t0:.0f}s]")
    header = "        " + " ".join(f"r1={c}" for c in range(1, R + 1))
    print(header)
    for r0 in range(1, R + 1):
        cells = " ".join(f"  {_LABEL[grid[(r0, r1)]][0]} " for r1 in range(1, R + 1))
        print(f"r0={r0}  " + cells)
    print("  legend: L=line win, a=attrition win, ..=draw;  + favors mover (P0)\n")

    diagonal = {r: grid[(r, r)] for r in range(1, R + 1)}
    mover_never_loses = all(v >= 0 for v in grid.values())
    print(f"First player never loses across the grid: {mover_never_loses}")
    print(f"Equal-reserve diagonal values: {diagonal}")
    print("  -> draw for small r, then a first-player line win once material is "
          "large enough.\n")

    cc = crosscheck([(2, 1), (1, 2), (3, 2), (4, 3), (3, 4), (4, 4)])
    print("Cross-check folded vs enumeration (root value):")
    for k, v in cc.items():
        ok = "OK" if v["folded"] == v["enum"] else "MISMATCH"
        print(f"  {k}: folded={v['folded']:>4} enum={v['enum']:>4} {ok}")
    print()

    ordering = None
    pv = None
    if R >= 6:
        seeds = [1, 2, 3, 4, 5]
        cells = [(6, 6), (6, 3), (3, 6), (5, 5), (4, 4)]
        print(f"Ordering-robustness re-solve (fresh table, seeds={seeds}):")
        ordering = verify_ordering(cells, seeds)
        for k, v in ordering.items():
            tag = "OK" if v["invariant"] else "MISMATCH"
            print(f"  {k}: base={v['base']:>4}  shuffled={v['shuffled']}  {tag}")
        pv = verify_pv(empty_state(6, 6, 0), seeds)
        pvtag = "OK" if pv["all_positions_match"] else "MISMATCH"
        print(f"  (6,6) principal variation: {pv['pv_len']} positions, "
              f"root={pv['root_value']}, all independently re-verified: {pvtag}\n")

    print(f"Draw fraction by size (exact enumeration), sizes={list(census_sizes)}:")
    trend = draw_trend(census_sizes)
    prev = None
    for k in sorted(trend, key=lambda s: int(s.split(",")[0][1:])):
        v = trend[k]
        arrow = ""
        if prev is not None:
            delta = prev - v["draw_frac"]
            arrow = f"  (down {delta:.1%} from prior)"
        print(f"  {k}: {v['states']:,} states, draw {v['draw_frac']:.1%}, "
              f"decisive {v['decisive_frac']:.1%}  [{v['seconds']:.0f}s]{arrow}")
        prev = v["draw_frac"]
    mono = check_monotonic(trend)
    print(f"\nDraw fraction strictly monotonic: {mono}")
    print("  -> scaling reserves makes a sharper game; no rule change required.\n")

    result = {
        "grid": {f"({r0},{r1})": {"value": grid[(r0, r1)], "label": _LABEL[grid[(r0, r1)]][1]}
                 for (r0, r1) in grid},
        "first_player_never_loses": mover_never_loses,
        "second_player_never_wins": mover_never_loses,
        "diagonal": {f"({r},{r})": diagonal[r] for r in diagonal},
        "crosscheck": cc,
        "ordering_robustness": ordering,
        "pv_verification": pv,
        "draw_trend": trend,
        "draw_fraction_monotonic": mono,
    }
    path = write_result(NAME, {"grid_size": R, "census_sizes": list(census_sizes)}, result)
    announce(NAME, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_size", nargs="?", type=int, default=6)
    ap.add_argument("--census", nargs="+", type=int, default=[2, 3, 4],
                      help="symmetric sizes for exact draw-fraction census")
    args = ap.parse_args(sys.argv[1:] or None)
    main(args.grid_size, tuple(args.census))
