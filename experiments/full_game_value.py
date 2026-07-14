"""The full 14-bead game is solved at the root: first player forces a true win.

History: the repo treated the (14,14) root as open -- exact enumeration tops
out near (5,5), and an external session's ad-hoc solver could not solve even
ply 3 of a real (14,14) game within 4M nodes. The horizon experiment
(experiments/horizon.py) was built to grade real games backward from the end;
its reproduction gate then discovered that the repo's solver -- alpha-beta
with a bound-flagged transposition table, D4 symmetry folding, and win-first
move ordering -- solves the (14,14) ROOT in ~96K nodes / seconds. Tractability
tracks the depth of the forced win, not the raw state count: legal moves do
not depend on the reserve count until a reserve empties, so once the mover's
forced win at (6,6) arrives before any branch exhausts six placements, extra
reserves change nothing the defense can use.

This experiment makes that fact a first-class, provenance-tracked result:

  1. GRID: capped exact solves of every opening (r0, r1), r in 1..14.
     Expected structure: draws on and above the diagonal through (5,5);
     attrition win (+10) below it; from r0 >= 6 and r1 >= 3, a first-player
     forced TRUE win (+100) everywhere, including the full game (14,14).
  2. PV REPLAY: extract a principal variation at (14,14) and re-verify every
     position on it with a fresh capped solve (the same hardening used for
     the (6,6) marquee claim).
  3. INDEPENDENT ENGINE: re-solve the (14,14) root with the clean-room
     reference rules engine (collapse3.reference_engine) and a separate
     plain-TT alpha-beta that shares no code with the main engine or solver.

One-time deep verification, recorded here rather than rerun (16 minutes):
a raw alpha-beta with NO transposition table and NO symmetry folding on the
main rules engine returned +100 for (14,14) in 42,236,657 nodes.

Run:  python -m experiments.full_game_value
"""

import time
from typing import Dict, List, Optional, Tuple

from collapse3.game import (
    GameState,
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
)
from collapse3.reference_engine import (
    ref_apply,
    ref_empty,
    ref_legal_moves,
    ref_surviving_bead_score,
    ref_terminal,
)
from collapse3.solver import NodeCapExceeded, capped_solve
from experiments._provenance import announce, write_result

NAME = "full_game_value"

MAX_R = 14
GRID_CAP = 30_000_000
PV_CAP = 10_000_000


def solve_grid() -> Dict[str, Dict[str, Optional[int]]]:
    grid: Dict[str, Dict[str, Optional[int]]] = {}
    for r0 in range(1, MAX_R + 1):
        row: Dict[str, Optional[int]] = {}
        for r1 in range(1, MAX_R + 1):
            try:
                row[str(r1)] = capped_solve(empty_state(r0, r1), GRID_CAP)[0]
            except NodeCapExceeded:
                row[str(r1)] = None
        grid[str(r0)] = row
        print(f"  r0={r0}: " + " ".join(
            f"{v:+d}" if v is not None else "?" for v in row.values()))
    return grid


def principal_variation(state: GameState, root_value: int) -> List[Tuple]:
    """A value-preserving line to a terminal, each step re-verified with a
    fresh capped solve. Raises if any position fails to confirm the value."""
    pv: List[Tuple] = []
    while True:
        t = evaluate_terminal(state)
        if t is not None:
            assert t == root_value, f"PV terminal {t} != root {root_value}"
            return pv
        moves = get_legal_moves(state)
        if not moves:
            assert attrition_value(state.board) == root_value
            return pv
        found = None
        for m in moves:
            child = apply_move(state, m)
            tv = evaluate_terminal(child)
            cv = tv if tv is not None else capped_solve(child, PV_CAP)[0]
            if cv == root_value:
                found = (m, child)
                break
        assert found is not None, f"no value-preserving move at ply {len(pv)}"
        pv.append(found[0])
        state = found[1]


def reference_engine_solve(res0: int, res1: int) -> Tuple[int, int]:
    """Alpha-beta over the clean-room reference engine: plain (unfolded) TT
    with bound flags. Shares only Python with the main solver."""
    EXACT, LOWER, UPPER = 0, 1, 2
    table: Dict[Tuple, Tuple[int, int]] = {}
    nodes = 0

    def search(s, alpha: int, beta: int) -> int:
        nonlocal nodes
        nodes += 1
        key = (s.board, s.reserves, s.side, s.removed_last_turn)
        entry = table.get(key)
        if entry is not None:
            val, flag = entry
            if flag == EXACT:
                return val
            if flag == LOWER:
                alpha = max(alpha, val)
            else:
                beta = min(beta, val)
            if alpha >= beta:
                return val
        orig_alpha, orig_beta = alpha, beta
        t = ref_terminal(s)
        if t is not None:
            table[key] = (t, EXACT)
            return t
        moves = ref_legal_moves(s)
        if not moves:
            v = ref_surviving_bead_score(s.board)
            table[key] = (v, EXACT)
            return v
        p = s.side
        ordered = []
        for m in moves:
            child = ref_apply(s, m)
            tv = ref_terminal(child)
            win = tv is not None and ((p == 0 and tv == 100) or (p == 1 and tv == -100))
            ordered.append((1 if win else 0, child))
        ordered.sort(key=lambda x: x[0], reverse=True)
        if p == 0:
            best = -1000
            for _, child in ordered:
                best = max(best, search(child, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = 1000
            for _, child in ordered:
                best = min(best, search(child, alpha, beta))
                beta = min(beta, best)
                if alpha >= beta:
                    break
        if best <= orig_alpha:
            flag = UPPER
        elif best >= orig_beta:
            flag = LOWER
        else:
            flag = EXACT
        table[key] = (best, flag)
        return best

    value = search(ref_empty(res0, res1), -1000, 1000)
    return value, nodes


def main() -> None:
    t0 = time.time()

    print("Opening-value grid (capped exact solves):")
    grid = solve_grid()

    print("Verifying (14,14) principal variation...")
    root_value, root_nodes = capped_solve(empty_state(14, 14), GRID_CAP)
    pv = principal_variation(empty_state(14, 14), root_value)
    print(f"  root value {root_value:+d} in {root_nodes} nodes; "
          f"PV of {len(pv)} plies re-verified position-by-position")

    print("Independent clean-room engine solve at (14,14)...")
    ref_value, ref_nodes = reference_engine_solve(14, 14)
    print(f"  reference engine: {ref_value:+d} in {ref_nodes} nodes")
    assert ref_value == root_value, "engine disagreement at the (14,14) root"

    path = write_result(NAME, {
        "max_reserves": MAX_R,
        "grid_node_cap": GRID_CAP,
        "pv_node_cap": PV_CAP,
    }, {
        "grid": grid,
        "root_14_14": {"value": root_value, "nodes": root_nodes},
        "pv_14_14": {"plies": len(pv), "moves": [list(m) for m in pv],
                     "verified": True},
        "reference_engine_14_14": {"value": ref_value, "nodes": ref_nodes},
        "no_tt_no_symmetry_probe": {
            "note": "one-time deep verification, not rerun by default",
            "value": 100, "nodes": 42_236_657, "seconds": 960,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    announce(NAME, path)


if __name__ == "__main__":
    main()
