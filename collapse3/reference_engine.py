"""Clean-room Collapse3 rules engine (reference implementation).

Written independently from :mod:`collapse3.game`, following ``rules.md`` only.
This module must not import from ``collapse3.game``; differential tests convert
states at the boundary and compare legal moves, transitions, and terminal
scores. Any disagreement means a rules bug in one engine or the other.

Board model: nine peg stacks (bottom-first tuples of 0/1), matching the
shipped engine's *semantics* but not its code paths — win lines are enumerated
from the rulebook's three categories; removal legality follows the five named
conditions in order.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# P0-perspective terminal scores (same convention as the shipped engine).
_TRUE_WIN = 100
_TRUE_LOSS = -100
_ATTRITION_WIN = 10
_ATTRITION_LOSS = -10
_DRAW = 0

Board = Tuple[Tuple[int, ...], ...]
Move = Tuple

# Base-grid collinear triples (rows, columns, diagonals on the 3×3 peg layout).
_BASE_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


@dataclass(frozen=True)
class RefState:
    """Minimal Markov state: board, reserves, side to move, removal cooldown."""

    board: Board
    reserves: Tuple[int, int]
    side: int                       # 0 or 1
    removed_last_turn: Tuple[bool, bool]


def ref_empty(res0: int, res1: int, side: int = 0) -> RefState:
    empty = tuple(tuple() for _ in range(9))
    return RefState(empty, (res0, res1), side, (False, False))


def _bead(board: Board, peg: int, level: int) -> Optional[int]:
    stack = board[peg]
    if level < 0 or level >= len(stack):
        return None
    return stack[level]


def _line_complete(board: Board, player: int, cells: Sequence[Tuple[int, int]]) -> bool:
    for peg, level in cells:
        if _bead(board, peg, level) != player:
            return False
    return True


def ref_has_line(board: Board, player: int) -> bool:
    """Three-in-a-row: vertical, flat (any level), or staircase (rules.md §Winning)."""
    for peg in range(9):
        if _line_complete(board, player, ((peg, 0), (peg, 1), (peg, 2))):
            return True
    for level in range(3):
        for a, b, c in _BASE_LINES:
            if _line_complete(board, player, ((a, level), (b, level), (c, level))):
                return True
    for a, b, c in _BASE_LINES:
        if _line_complete(board, player, ((a, 0), (b, 1), (c, 2))):
            return True
        if _line_complete(board, player, ((a, 2), (b, 1), (c, 0))):
            return True
    return False


def ref_surviving_bead_score(board: Board) -> int:
    p0 = sum(stack.count(0) for stack in board)
    p1 = sum(stack.count(1) for stack in board)
    if p0 > p1:
        return _ATTRITION_WIN
    if p1 > p0:
        return _ATTRITION_LOSS
    return _DRAW


def ref_terminal(state: RefState) -> Optional[int]:
    """Terminal score from P0's perspective, or None if play continues."""
    p0_line = ref_has_line(state.board, 0)
    p1_line = ref_has_line(state.board, 1)
    if p0_line or p1_line:
        if p0_line and p1_line:
            mover = 1 - state.side
            return _TRUE_WIN if mover == 0 else _TRUE_LOSS
        return _TRUE_WIN if p0_line else _TRUE_LOSS
    if state.reserves[0] == 0 and state.reserves[1] == 0:
        return ref_surviving_bead_score(state.board)
    return None


def ref_legal_moves(state: RefState) -> List[Move]:
    moves: List[Move] = []
    me = state.side
    opp = 1 - me
    board = state.board

    # Placement (rules.md §Action 1).
    if state.reserves[me] > 0:
        for peg in range(9):
            if len(board[peg]) < 3:
                moves.append(("place", peg))

    # Removal — all five conditions (rules.md §Conditions for a Legal Removal).
    if not state.removed_last_turn[me]:                    # 1. Cooldown
        heights = [len(board[p]) for p in range(9)]
        tallest = max(heights) if heights else 0
        if tallest >= 2:                                   # 2. Singleton Immunity
            for peg in range(9):
                if heights[peg] != tallest:                # 3. Tallest Stack
                    continue
                stack = board[peg]
                if not stack or stack[-1] != opp:          # 4. Capping Rule
                    continue
                for level, bead in enumerate(stack):       # 5. Target Rule
                    if bead == opp:
                        moves.append(("remove", peg, level))
    return moves


def ref_apply(state: RefState, move: Move) -> RefState:
    board = [list(stack) for stack in state.board]
    reserves = list(state.reserves)
    removed_last = list(state.removed_last_turn)
    me = state.side

    if move[0] == "place":
        peg = move[1]
        board[peg].append(me)
        reserves[me] -= 1
        removed_last[me] = False
    elif move[0] == "remove":
        peg, level = move[1], move[2]
        stack = board[peg]
        board[peg] = stack[:level] + stack[level + 1:]
        removed_last[me] = True
    else:
        raise ValueError(f"unknown move: {move!r}")

    return RefState(
        board=tuple(tuple(s) for s in board),
        reserves=tuple(reserves),
        side=1 - me,
        removed_last_turn=tuple(removed_last),
    )


def ref_end_if_stuck(state: RefState) -> Optional[int]:
    """Repo rule: no legal action -> immediate attrition end."""
    if ref_terminal(state) is not None:
        return ref_terminal(state)
    if not ref_legal_moves(state):
        return ref_surviving_bead_score(state.board)
    return None
