"""Lock the board notation (collapse3/render.py) to NOTATION.md byte-for-byte.

If the renderer changes, or NOTATION.md drifts from it, this test fails -- the
documented notation can never silently diverge from what the code prints.
"""

from pathlib import Path

from collapse3.game import GameState, empty_state, apply_move
from collapse3.render import render_board, render_state

NOTATION = Path(__file__).resolve().parent.parent / "NOTATION.md"

# The single worked example used throughout NOTATION.md.
EXAMPLE = GameState(
    board=((), (1,), (), (0,), (0, 1), (0,), (1, 1), (), ()),
    res=(10, 10),
    turn=1,
    cooldown=(False, False),
)

EMPTY_RENDER = """\
┌───┬───┬───┐
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
└───┴───┴───┘"""

EXAMPLE_RENDER = """\
┌───┬───┬───┐
│ - │ - │ - │
│ - │ - │ - │
│ - │ X │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ X │ - │
│ O │ O │ O │
├───┼───┼───┤
│ - │ - │ - │
│ X │ - │ - │
│ X │ - │ - │
└───┴───┴───┘"""

FOOTER = "reserves O=10 X=10  turn=X  cooldown O=False X=False"

# Labeled (orientation-annotated) renderings — what NOTATION.md actually shows.
EMPTY_LABELED = """\
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ - │ - │ - │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘"""

EXAMPLE_LABELED = """\
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ - │ X │ - │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ X │ - │
      │ O │ O │ O │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ X │ - │ - │
      │ X │ - │ - │
      └───┴───┴───┘"""


def test_empty_board_render():
    assert render_board(tuple(() for _ in range(9))) == EMPTY_RENDER


def test_example_board_render():
    assert render_board(EXAMPLE.board) == EXAMPLE_RENDER


def test_labeled_render_matches():
    assert render_board(tuple(() for _ in range(9)), labeled=True) == EMPTY_LABELED
    assert render_board(EXAMPLE.board, labeled=True) == EXAMPLE_LABELED


def test_example_state_footer():
    assert render_state(EXAMPLE) == f"{EXAMPLE_RENDER}\n{FOOTER}"


def test_notation_md_contains_the_rendered_blocks():
    """The labeled blocks and the footer line must appear verbatim in the doc."""
    doc = NOTATION.read_text()
    assert EMPTY_LABELED in doc
    assert EXAMPLE_LABELED in doc
    assert FOOTER in doc


def test_gravity_is_visible_bottom_up():
    """A placed bead occupies Level 1 (bottom row of its cell), not the top."""
    s = apply_move(empty_state(4, 4), ("place", 0))
    lines = render_board(s.board).splitlines()
    # Peg 0 is the top-left cell: grid rows 1-3 of the render are lines 1-3;
    # its Level 1 is the third of those (bottom), which must hold the O.
    assert lines[3].startswith("│ O ")   # Level 1 row of the top-left cell
    assert lines[1].startswith("│ - ")   # Level 3 row is still empty
