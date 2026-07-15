"""Collapse3 rules engine.

Pure, dependency-free implementation of the game defined in ``rules.md``.
This module owns the state representation, move generation, move application,
win detection, and terminal evaluation. The solver, oracle, agents, and metrics
all build on top of it so there is a single source of truth for the rules.
Cross-validated against :mod:`collapse3.reference_engine` — a clean-room second
implementation written from ``rules.md`` alone (see ``tests/test_reference_engine.py``).

Score convention (used everywhere downstream):
    +100 / -100  true 3-in-a-row win for P0 / P1
    + 10 / - 10  exhaustion (surviving-bead) tiebreaker win for P0 / P1
        0        draw
All scores are expressed from P0's global perspective (P0 maximizes,
P1 minimizes). Use :func:`orient` to view a score from a given player's seat.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

Board = Tuple[Tuple[int, ...], ...]
Move = Tuple

# Ordinal outcome ladder (from the acting player's perspective, after orient()).
# Used for value-based, distribution-independent regret: the "value drop" of a
# move is how far it slides the outcome down this ladder under optimal play.
TRUE_WIN = 100
ATTRITION_WIN = 10
DRAW = 0
ATTRITION_LOSS = -10
TRUE_LOSS = -100


@dataclass(frozen=True)
class GameState:
    board: Board                 # 9 pegs; each a tuple of beads (0 or 1), bottom first
    res: Tuple[int, int]         # remaining Reserve for (P0, P1)
    turn: int                    # whose turn it is: 0 or 1
    cooldown: Tuple[bool, bool]  # True if that player's LAST action was a Removal


# The 8 base grid lines; all winning lines derive from these plus verticals.
WINNING_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)

# The 8 symmetries of the 3x3 base grid (dihedral group D4), as peg-index
# permutations. They permute which peg is which; each peg's vertical stack is
# untouched (levels are orthogonal to the grid plane) and res/turn/cooldown are
# symmetry-invariant. Winning lines map among themselves, so the game value is
# invariant under all 8 -- used to canonicalize the transposition-table key.
SYMMETRIES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),  # identity
    (6, 3, 0, 7, 4, 1, 8, 5, 2),  # rotate 90
    (8, 7, 6, 5, 4, 3, 2, 1, 0),  # rotate 180
    (2, 5, 8, 1, 4, 7, 0, 3, 6),  # rotate 270
    (2, 1, 0, 5, 4, 3, 8, 7, 6),  # flip horizontal
    (6, 7, 8, 3, 4, 5, 0, 1, 2),  # flip vertical
    (0, 3, 6, 1, 4, 7, 2, 5, 8),  # transpose (main diagonal)
    (8, 5, 2, 7, 4, 1, 6, 3, 0),  # anti-transpose (anti-diagonal)
)


def empty_state(res0: int, res1: int, turn: int = 0) -> GameState:
    """A cleared board with the given reserves; P0 to move by default."""
    return GameState(tuple(tuple() for _ in range(9)), (res0, res1), turn, (False, False))


def canonical_board(board: Board) -> Board:
    """Lexicographically smallest board over all 8 grid symmetries."""
    return min(tuple(board[p] for p in perm) for perm in SYMMETRIES)


def tt_key(state: GameState) -> Tuple:
    """Canonical transposition-table key, folding the 8 grid symmetries."""
    return (canonical_board(state.board), state.res, state.turn, state.cooldown)


def orient(score: int, player: int) -> int:
    """Re-express a P0-perspective score from ``player``'s seat.

    A higher oriented value is always better for ``player``.
    """
    return score if player == 0 else -score


# -----------------------------------------------------------------------------
# Win detection & terminal evaluation
# -----------------------------------------------------------------------------
def has_win(player: int, board: Board) -> bool:
    # Vertical: a full peg of one color.
    for peg in board:
        if len(peg) == 3 and peg[0] == player and peg[1] == player and peg[2] == player:
            return True

    # Flat horizontal (rows/cols/diagonals) on each level z.
    for z in range(3):
        for p1, p2, p3 in WINNING_LINES:
            if (len(board[p1]) > z and board[p1][z] == player and
                    len(board[p2]) > z and board[p2][z] == player and
                    len(board[p3]) > z and board[p3][z] == player):
                return True

    # Staircase: bead levels 0,1,2 (or 2,1,0) along a base line. The geometric
    # middle peg necessarily holds the middle level. Bead positions only --
    # total stack heights are irrelevant.
    for p1, p2, p3 in WINNING_LINES:
        if (len(board[p1]) > 0 and board[p1][0] == player and
                len(board[p2]) > 1 and board[p2][1] == player and
                len(board[p3]) > 2 and board[p3][2] == player):
            return True
        if (len(board[p1]) > 2 and board[p1][2] == player and
                len(board[p2]) > 1 and board[p2][1] == player and
                len(board[p3]) > 0 and board[p3][0] == player):
            return True

    return False


def attrition_value(board: Board) -> int:
    """Exhaustion tiebreaker: +10 P0, -10 P1, 0 draw."""
    p0 = sum(peg.count(0) for peg in board)
    p1 = sum(peg.count(1) for peg in board)
    if p0 > p1:
        return ATTRITION_WIN
    if p1 > p0:
        return ATTRITION_LOSS
    return DRAW


def evaluate_terminal(state: GameState) -> Optional[int]:
    """Terminal score, or ``None`` if the game continues.

    Handles the Oops rule (a Removal that aligns only the opponent's beads
    gives the opponent the win) and the Simultaneous rule (if both players have
    a line after one action, the player who just moved wins the tie).
    """
    p0_win = has_win(0, state.board)
    p1_win = has_win(1, state.board)

    if p0_win or p1_win:
        if p0_win and p1_win:
            just_moved = 1 - state.turn
            return TRUE_WIN if just_moved == 0 else TRUE_LOSS
        return TRUE_WIN if p0_win else TRUE_LOSS

    # Rulebook tiebreaker: once both reserves are empty with no line, the game
    # ends immediately by surviving-bead count (no further removals occur).
    if state.res[0] == 0 and state.res[1] == 0:
        return attrition_value(state.board)

    return None


# -----------------------------------------------------------------------------
# Move generation & application
# -----------------------------------------------------------------------------
# Variant hook: which pegs accept a placement. Default is all 9 (standard
# Collapse3). The three-peg sibling variant restricts placements to the single
# row (0, 1, 2) -- a pure move-legality restriction, not a forked engine. Every
# other mechanic (stacking, gravity, removal legality, cooldown, Oops, line/
# attrition wins) is untouched; pegs 3-8 simply stay empty, so their winning
# lines can never fire. Removal legality already self-restricts to occupied
# pegs, so only placements need gating. See ``placement_pegs`` below.
PLACEMENT_PEGS: Tuple[int, ...] = tuple(range(9))


@contextmanager
def placement_pegs(pegs: Tuple[int, ...]) -> Iterator[None]:
    """Temporarily restrict which pegs accept a placement (variant scope).

    Used by the three-peg sibling variant: ``with placement_pegs((0, 1, 2)):``
    makes every ``get_legal_moves`` call in the block emit placements only on
    the first row. The restriction is a global the enumerator/solver read, so it
    threads through :func:`get_legal_moves` without changing any call site.
    """
    global PLACEMENT_PEGS
    prev = PLACEMENT_PEGS
    PLACEMENT_PEGS = tuple(pegs)
    try:
        yield
    finally:
        PLACEMENT_PEGS = prev


def get_legal_moves(state: GameState) -> List[Move]:
    moves: List[Move] = []
    p = state.turn
    opp = 1 - p
    board = state.board

    # Placements (restricted to PLACEMENT_PEGS for variants; all 9 by default).
    if state.res[p] > 0:
        for peg_idx in PLACEMENT_PEGS:
            if len(board[peg_idx]) < 3:
                moves.append(('place', peg_idx))

    # Removals (all five legality conditions from rules.md).
    if not state.cooldown[p]:                       # 1. Cooldown
        heights = [len(peg) for peg in board]
        max_h = max(heights)
        if max_h >= 2:                              # 2. Singleton Immunity
            for peg_idx in range(9):
                if heights[peg_idx] == max_h:       # 3. Tallest Stack
                    peg = board[peg_idx]
                    if peg[-1] == opp:              # 4. Capping Rule
                        for z, bead in enumerate(peg):
                            if bead == opp:         # 5. Any opponent bead
                                moves.append(('remove', peg_idx, z))
    return moves


def apply_move(state: GameState, move: Move) -> GameState:
    board = list(state.board)
    res = list(state.res)
    cooldown = list(state.cooldown)
    p = state.turn

    kind = move[0]
    if kind == 'place':
        peg_idx = move[1]
        board[peg_idx] = board[peg_idx] + (p,)
        res[p] -= 1
        cooldown[p] = False
    elif kind == 'remove':
        peg_idx, z = move[1], move[2]
        peg = board[peg_idx]
        board[peg_idx] = peg[:z] + peg[z + 1:]      # gravity cascade
        cooldown[p] = True
    else:
        raise ValueError(f"unknown move kind: {move!r}")

    return GameState(tuple(board), tuple(res), 1 - p, tuple(cooldown))
