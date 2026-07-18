"""Plain-text board notation for Collapse3 (see NOTATION.md).

Renders a :class:`~collapse3.game.GameState` as a vertically-stacked ASCII grid
that mirrors the physical board: the 3x3 grid of pegs, each peg drawn as three
level-slots with Level 3 on top and Level 1 on the bottom, so gravity reads
correctly (no bead ever floats above an empty slot). Player 0 is ``O``,
player 1 is ``X``, an empty slot is ``-``.

Pure standard library; the engine never imports this (display only). The
example in NOTATION.md is locked byte-for-byte to this function by
``tests/test_render.py`` so the documented notation can never drift from what
the code produces.
"""

from typing import Tuple

from .game import Board, GameState

GLYPHS = {0: "O", 1: "X"}
EMPTY = "-"


def render_board(board: Board) -> str:
    """ASCII rendering of the nine pegs (board only, no reserves/turn)."""
    # Each peg -> three chars, index 0 = Level 1 (bottom) ... index 2 = Level 3.
    def slot(peg: Tuple[int, ...], level: int) -> str:
        return GLYPHS[peg[level]] if level < len(peg) else EMPTY

    top = "┌───┬───┬───┐"
    mid = "├───┼───┼───┤"
    bot = "└───┴───┴───┘"

    lines = [top]
    for grid_row in range(3):
        pegs = [board[grid_row * 3 + col] for col in range(3)]
        for level in (2, 1, 0):                       # Level 3 printed first (top)
            cells = " │ ".join(slot(peg, level) for peg in pegs)
            lines.append(f"│ {cells} │")
        lines.append(mid if grid_row < 2 else bot)
    return "\n".join(lines)


def render_state(state: GameState) -> str:
    """Board plus a one-line status footer (reserves, turn, cooldown)."""
    board = render_board(state.board)
    footer = (f"reserves O={state.res[0]} X={state.res[1]}  "
              f"turn={GLYPHS[state.turn]}  "
              f"cooldown O={state.cooldown[0]} X={state.cooldown[1]}")
    return f"{board}\n{footer}"
