"""Lock the 30-second video demos to the engine (30SECONDS.md).

Every board in the video script is rebuilt from the rules engine, checked for
the claimed outcome (win / Oops / clever collapse), rendered, and asserted to
appear verbatim in 30SECONDS.md. If a rule or the renderer changes, or the doc
drifts, this fails -- the video can never show an illegal or wrong position.
"""

from pathlib import Path

from collapse3.game import (
    GameState,
    apply_move,
    evaluate_terminal,
    get_legal_moves,
    has_win,
)
from collapse3.render import render_board

DOC = (Path(__file__).resolve().parent.parent / "30SECONDS.md").read_text()


def _state(board, res=(5, 5), turn=0, cd=(False, False)):
    return GameState(board, res, turn, cd)


def _shown(board):
    """The full labeled board appears verbatim in the doc.

    Every board is rendered with orientation labels and the collapse demos are
    stacked (BEFORE above AFTER), so each block is present byte-for-byte.
    """
    return render_board(board, labeled=True) in DOC


def test_winning_positions_are_wins_and_shown():
    for board in (
        ((0,), (0,), (0,), (), (), (), (), (), ()),          # flat, level 1
        ((0, 0, 0), (), (), (), (), (), (), (), ()),         # vertical
        ((0,), (1, 0), (1, 1, 0), (), (), (), (), (), ()),   # staircase
    ):
        assert has_win(0, board)
        assert _shown(board)


def test_simple_collapse_shows_gravity_no_win():
    s = _state(((), (), (), (), (1, 0, 1), (), (), (), ()))
    mv = ("remove", 4, 0)
    assert mv in get_legal_moves(s)
    after = apply_move(s, mv)
    assert evaluate_terminal(after) is None
    assert _shown(s.board) and _shown(after.board)


def test_oops_hands_opponent_the_win():
    s = _state(((0, 1), (1, 0, 1), (1, 1), (), (), (), (), (), ()))
    mv = ("remove", 1, 0)
    assert mv in get_legal_moves(s)
    after = apply_move(s, mv)
    assert has_win(1, after.board) and not has_win(0, after.board)
    assert evaluate_terminal(after) == -100          # TRUE_LOSS: X wins
    assert _shown(s.board) and _shown(after.board)


def test_clever_collapse_wins_on_the_spot():
    s = _state(((0,), (1, 0, 1), (0,), (), (), (), (), (), ()))
    mv = ("remove", 1, 0)
    assert mv in get_legal_moves(s)
    after = apply_move(s, mv)
    assert has_win(0, after.board) and not has_win(1, after.board)
    assert evaluate_terminal(after) == 100           # TRUE_WIN: O wins
    assert _shown(s.board) and _shown(after.board)
