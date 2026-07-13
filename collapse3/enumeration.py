"""Exhaustive reachable-state solving over the *un-folded* state space.

The main solver (:mod:`collapse3.solver`) folds the 8 grid symmetries into a
canonical transposition key -- ideal for computing a single value fast. The
representation/aliasing analyses, by contrast, must reason about *concrete*
states grouped by a lossy observation, so they need every reachable state
enumerated explicitly (no symmetry folding).

Everything here uses the shipped rules: **immediate game-end on no legal
action** (there is no pass). A non-terminal state with no legal move is scored
by the surviving-bead (attrition) count.
"""

from typing import Dict, List, Set

from .game import (
    GameState,
    Move,
    apply_move,
    attrition_value,
    evaluate_terminal,
    get_legal_moves,
    orient,
)


def reachable_states(root: GameState) -> Set[GameState]:
    """Every state reachable from ``root`` under legal play (un-folded)."""
    seen: Set[GameState] = {root}
    stack: List[GameState] = [root]
    while stack:
        s = stack.pop()
        if evaluate_terminal(s) is not None:
            continue
        for m in get_legal_moves(s):        # [] at a stuck non-terminal state
            c = apply_move(s, m)
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def solve_all(root: GameState) -> Dict[GameState, int]:
    """Exact P0-perspective value of every reachable state.

    Immediate-end semantics: terminal states use :func:`evaluate_terminal`; a
    non-terminal state with no legal move is scored by attrition.
    """
    memo: Dict[GameState, int] = {}

    def value(s: GameState) -> int:
        cached = memo.get(s)
        if cached is not None:
            return cached
        t = evaluate_terminal(s)
        if t is not None:
            memo[s] = t
            return t
        moves = get_legal_moves(s)
        if not moves:
            v = attrition_value(s.board)   # no pass: game ends here
            memo[s] = v
            return v
        vals = [value(apply_move(s, m)) for m in moves]
        v = max(vals) if s.turn == 0 else min(vals)
        memo[s] = v
        return v

    # Build the state set, then value children-before-parents. Game length is
    # bounded (placements are finite and removals strictly shrink the board), so
    # ordinary recursion depth is safe.
    order: List[GameState] = []
    seen: Set[GameState] = {root}
    stack: List[GameState] = [root]
    while stack:
        s = stack.pop()
        order.append(s)
        if evaluate_terminal(s) is not None:
            continue
        for m in get_legal_moves(s):
            c = apply_move(s, m)
            if c not in seen:
                seen.add(c)
                stack.append(c)
    for s in reversed(order):
        value(s)
    return memo


def wdl(value: int, player: int) -> int:
    """Collapse an exact value to win/draw/loss units from ``player``'s seat."""
    u = orient(value, player)
    return 1 if u > 0 else (-1 if u < 0 else 0)


def action_values(memo: Dict[GameState, int], state: GameState, unit: str = "wdl") -> Dict[Move, int]:
    """Value of each legal move from the mover's seat.

    ``unit='wdl'`` gives win/draw/loss (+1/0/-1); ``unit='raw'`` gives the
    oriented solver-unit value.
    """
    mover = state.turn
    out: Dict[Move, int] = {}
    for m in get_legal_moves(state):
        v = memo[apply_move(state, m)]
        out[m] = wdl(v, mover) if unit == "wdl" else orient(v, mover)
    return out
