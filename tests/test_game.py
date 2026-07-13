"""Rules-engine tests: symmetries, win detection, Oops/Simultaneous, moves."""

from collapse3.game import (
    GameState,
    SYMMETRIES,
    WINNING_LINES,
    apply_move,
    evaluate_terminal,
    get_legal_moves,
    has_win,
)

E = tuple()


def board(**kw):
    b = [E] * 9
    for k, v in kw.items():
        b[int(k[1:])] = tuple(v)
    return tuple(b)


def test_symmetries_are_line_preserving_permutations():
    lines = set(frozenset(l) for l in WINNING_LINES)
    for perm in SYMMETRIES:
        assert sorted(perm) == list(range(9))
        mapped = set(frozenset(perm[i] for i in l) for l in WINNING_LINES)
        assert mapped == lines


def test_win_detection_vertical_flat_staircase():
    assert has_win(0, board(p0=[0, 0, 0]))                       # vertical
    assert not has_win(1, board(p0=[0, 0, 0]))
    assert has_win(1, board(p0=[1], p1=[1], p2=[1]))             # flat L0
    assert has_win(0, board(p0=[1, 1, 0], p1=[1, 1, 0], p2=[1, 1, 0]))  # flat L2
    assert has_win(0, board(p0=[0], p1=[1, 0], p2=[1, 1, 0]))    # staircase up
    assert has_win(0, board(p0=[1, 1, 0], p1=[1, 0], p2=[0]))    # staircase down
    # middle peg at wrong level is not a staircase
    assert not has_win(0, board(p0=[0], p1=[0], p2=[1, 1, 0]))


def test_oops_rule_opponent_wins_from_your_move():
    # P0 just moved (turn now 1); only P1 has a line -> P1 wins.
    child = GameState(board(p0=[1], p1=[1], p2=[1]), (1, 1), 1, (False, True))
    assert evaluate_terminal(child) == -100


def test_simultaneous_rule_mover_wins_tie():
    b = board(p0=[0, 0, 0], p3=[1], p4=[1], p5=[1])
    assert has_win(0, b) and has_win(1, b)
    assert evaluate_terminal(GameState(b, (1, 1), 1, (False, False))) == 100    # P0 moved
    assert evaluate_terminal(GameState(b, (1, 1), 0, (False, False))) == -100   # P1 moved


def test_removal_conditions_and_gravity():
    # Tallest stack height 2, opponent (P1) on top of a P0/P1 stack -> removable.
    b = board(p0=[0, 1], p1=[0])
    st = GameState(b, (5, 5), 0, (False, False))
    moves = get_legal_moves(st)
    assert ("remove", 0, 1) in moves           # remove opp top
    assert all(mv[0] != "remove" or mv[1] != 1 for mv in moves)  # peg1 is a singleton
    # Cooldown blocks a second consecutive removal.
    after = apply_move(st, ("remove", 0, 1))
    assert after.cooldown[0] is True
    assert after.board[0] == (0,)              # gravity: bottom P0 remains


def test_reserves_empty_is_terminal_by_attrition():
    b = board(p0=[0, 0], p1=[1])   # P0 has 2, P1 has 1
    st = GameState(b, (0, 0), 0, (False, False))
    assert evaluate_terminal(st) == 10   # P0 wins by surviving-bead count
