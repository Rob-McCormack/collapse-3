#!/usr/bin/env python3
"""Collapse3 exact solver.

Minimax + alpha-beta + transposition table with bound flags (EXACT/LOWER/UPPER).
The rules engine lives in :mod:`collapse3.game`; this module only implements the
search. It exposes :func:`game_value` (the exact P0-perspective value of a state
under optimal play), which the oracle uses to produce ground-truth labels.

Design notes preserved from earlier versions:
  * The TT stores bound flags, so values cut off by alpha-beta windows are never
    served as exact results. (This was the correctness-critical fix.)
  * No-legal-action ends the game immediately by the surviving-bead (attrition)
    count, matching the rulebook. There is no pass.
  * Weighted terminal scores (+/-100 true win, +/-10 tiebreaker, 0 draw) are
    deliberate: the solver prefers winning by alignment over attrition, and
    attrition loss over alignment loss. Signs are preserved, so the
    game-theoretic winner is unaffected.
"""

import random
import time
from typing import Dict, Optional, Tuple

from .game import (
    GameState,
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    tt_key,
)

EXACT, LOWER, UPPER = 0, 1, 2

# Module-level search cache. Keyed by canonical (symmetry-folded) state.
transposition_table: Dict[Tuple, Tuple[int, int]] = {}
nodes_visited = 0

# Optional RNG to perturb move-exploration order. When set, the base move list
# is shuffled before the (stable) win-first sort, so alpha-beta explores equal-
# priority moves in a different order. It cannot change the exact value; it is a
# robustness probe for ordering-sensitive alpha-beta / transposition-table bugs.
_order_rng: Optional[random.Random] = None


def reset_search_state() -> None:
    global nodes_visited
    transposition_table.clear()
    nodes_visited = 0


def minimax(state: GameState, alpha: int = -1000, beta: int = 1000) -> int:
    """Exact P0-perspective value of ``state`` under optimal play."""
    global nodes_visited
    nodes_visited += 1
    orig_alpha, orig_beta = alpha, beta
    key = tt_key(state)  # canonical under the 8 grid symmetries

    entry = transposition_table.get(key)
    if entry is not None:
        val, flag = entry
        if flag == EXACT:
            return val
        if flag == LOWER:
            alpha = max(alpha, val)
        elif flag == UPPER:
            beta = min(beta, val)
        if alpha >= beta:
            return val

    terminal = evaluate_terminal(state)
    if terminal is not None:
        transposition_table[key] = (terminal, EXACT)
        return terminal

    # Rulebook: if the active player has no legal action, the game ends
    # immediately and is scored by the surviving-bead (attrition) count.
    moves = get_legal_moves(state)
    if not moves:
        val = attrition_value(state.board)
        transposition_table[key] = (val, EXACT)
        return val

    # Move ordering: immediate true wins first. A stable sort preserves the base
    # order within equal keys, so an optional shuffle perturbs exploration order.
    if _order_rng is not None:
        _order_rng.shuffle(moves)
    p = state.turn
    ordered = []
    for move in moves:
        child = apply_move(state, move)
        t = evaluate_terminal(child)
        winning = t is not None and ((p == 0 and t == 100) or (p == 1 and t == -100))
        ordered.append((1 if winning else 0, child))
    ordered.sort(key=lambda x: x[0], reverse=True)

    if p == 0:
        best = -1000
        for _, child in ordered:
            best = max(best, minimax(child, alpha, beta))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    else:
        best = 1000
        for _, child in ordered:
            best = min(best, minimax(child, alpha, beta))
            beta = min(beta, best)
            if alpha >= beta:
                break

    if best <= orig_alpha:
        flag = UPPER      # true value could be even lower
    elif best >= orig_beta:
        flag = LOWER      # true value could be even higher
    else:
        flag = EXACT
    transposition_table[key] = (best, flag)
    return best


def game_value(state: GameState, fresh: bool = False, order_seed: Optional[int] = None) -> int:
    """Exact P0-perspective value of ``state``.

    By default reuses the shared transposition table across calls (fast for
    batch analysis). Pass ``fresh=True`` to clear it first. Pass ``order_seed``
    to solve from a **fresh** table with move ordering shuffled by that seed --
    a robustness probe: the value must be invariant to exploration order.
    """
    global _order_rng
    if order_seed is not None:
        _order_rng = random.Random(order_seed)
        reset_search_state()
        try:
            return minimax(state)
        finally:
            _order_rng = None
    if fresh:
        reset_search_state()
    return minimax(state)


def solve_reserves(res0: int, res1: int) -> dict:
    reset_search_state()
    initial = empty_state(res0, res1)

    start = time.time()
    value = minimax(initial)
    elapsed = time.time() - start

    outcomes = {
        100: "P0 (true 3-in-a-row win)",
        -100: "P1 (true 3-in-a-row win)",
        10: "P0 (attrition tiebreaker win)",
        -10: "P1 (attrition tiebreaker win)",
        0: "Draw",
    }
    return {
        "score_value": value,
        "optimal_outcome": outcomes.get(value, f"unexpected value {value}"),
        "nodes_visited": nodes_visited,
        "table_size": len(transposition_table),
        "time_seconds": round(elapsed, 4),
    }


if __name__ == "__main__":
    print("Collapse3 Exact Solver (TT with bound flags)")
    print("=" * 60)
    for r in [1, 2, 3, 4]:
        print(f"Reserves = ({r}, {r}):")
        result = solve_reserves(r, r)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("-" * 30)
