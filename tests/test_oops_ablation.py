"""Guards for the Oops ablation: the rule is strategically inert but real.

Fast live checks on a hand-built Oops position and the (3,3) root, plus a record
guard that locks the exhaustive (4,4)/(5,5) census and the full-game ablation to
the committed results file.
"""

import json
import pathlib

import pytest

from collapse3.game import (
    TRUE_LOSS,
    GameState,
    apply_move,
    empty_state,
    evaluate_terminal,
    forbid_oops,
    get_legal_moves,
    has_win,
    is_oops,
    orient,
)
from collapse3.solver import game_value

RECORD = pathlib.Path(__file__).resolve().parents[1] / "results" / "oops_ablation_latest.json"

# A hand-built genuine Oops. P0 to move; removing the bottom P1 bead from peg 1
# drops the P1 above it to level 2, completing the ascending staircase
# peg0/level1 - peg1/level2 - peg2/level3 for P1 alone.
OOPS_BOARD = ((1,), (1, 0, 1), (0, 0, 1), (), (), (), (), (), ())
OOPS_STATE = GameState(OOPS_BOARD, (1, 1), 0, (False, False))
OOPS_MOVE = ('remove', 1, 0)


def test_hand_built_position_is_a_genuine_oops():
    # Not already won by anyone, and the removal is legal under the shipped rules.
    assert evaluate_terminal(OOPS_STATE) is None
    assert not has_win(0, OOPS_BOARD) and not has_win(1, OOPS_BOARD)
    assert OOPS_MOVE in get_legal_moves(OOPS_STATE)
    assert is_oops(OOPS_STATE, OOPS_MOVE)

    # The cascade lines up the opponent ALONE -- if it lined up both, the
    # Simultaneous rule would award the mover the win and this would not be an Oops.
    child = apply_move(OOPS_STATE, OOPS_MOVE)
    assert has_win(1, child.board) and not has_win(0, child.board)


def test_lemma1_an_oops_is_worth_exactly_true_loss():
    child = apply_move(OOPS_STATE, OOPS_MOVE)
    assert orient(evaluate_terminal(child), OOPS_STATE.turn) == TRUE_LOSS


def test_ablation_removes_exactly_the_oops_move():
    legal = get_legal_moves(OOPS_STATE)
    with forbid_oops():
        ablated = get_legal_moves(OOPS_STATE)
    assert set(legal) - set(ablated) == {OOPS_MOVE}


def test_forbid_oops_scope_is_restored():
    before = get_legal_moves(OOPS_STATE)
    with forbid_oops():
        pass
    assert get_legal_moves(OOPS_STATE) == before


def test_placements_are_never_oops():
    # Only a Removal can cascade, so a Placement can never be an Oops.
    for m in get_legal_moves(OOPS_STATE):
        if m[0] == 'place':
            assert not is_oops(OOPS_STATE, m)


def test_ablation_does_not_change_the_3_3_root():
    root = empty_state(3, 3)
    standard = game_value(root)
    with forbid_oops():
        ablated = game_value(root)
    assert standard == ablated == 0


@pytest.mark.skipif(not RECORD.exists(), reason="no committed oops_ablation record")
def test_recorded_census_and_ablation():
    rec = json.loads(RECORD.read_text())["results"]
    census, verdict = rec["census"], rec["verdict"]

    # Exhaustive counts: the mechanic proliferates with board size...
    assert census["(3,3)"]["states_with_oops_removal"] == 0
    assert census["(4,4)"]["states_with_oops_removal"] == 16
    assert census["(5,5)"]["states_with_oops_removal"] == 2368

    for size in ("(3,3)", "(4,4)", "(5,5)"):
        c = census[size]
        # ...but never becomes correct: every Oops is strictly worse than an
        # alternative, none is ever tied-optimal, and none is ever forced.
        assert c["oops_strictly_worse"] == c["oops_moves"], size
        assert c["oops_tied_optimal"] == 0, size
        assert c["oops_uniquely_optimal"] == 0, size
        assert c["states_all_legal_moves_oops"] == 0, size
        # Lemmas 1-2 verified over the whole state space.
        assert c["lemma1_true_loss_violations"] == 0, size
        assert c["lemma2_tied_implies_all_tied_violations"] == 0, size

    # Value-by-value ablation over the full state space at the two sizes where
    # holding two solved memos is affordable.
    for size in ("(3,3)", "(4,4)"):
        assert rec["ablation"][size]["value_disagreements"] == 0, size
        assert rec["ablation"][size]["ablated_states_not_in_standard"] == 0, size
    # Removing 16 Oops moves removes exactly 16 reachable (terminal) states.
    assert (rec["ablation"]["(4,4)"]["standard_states"]
            - rec["ablation"]["(4,4)"]["ablated_states"] == 16)

    # Root values identical with and without the rule, all the way to the full game.
    for size, row in rec["root_diagonal"].items():
        assert row["unchanged"] is True, size
    assert rec["root_diagonal"]["(14,14)"]["standard"] == 100

    assert verdict["oops_ever_strictly_best_move"] is False
    assert verdict["oops_ever_forced_all_legal_moves"] is False
    assert verdict["exact_values_unchanged_by_ablation"] is True
    assert verdict["full_game_root_unchanged"] is True


@pytest.mark.skipif(not RECORD.exists(), reason="no committed oops_ablation record")
def test_recorded_misere_ablation():
    """Relevance is objective-relative: the rule is inert for normal-play values
    and for the misere yes/no theorems, yet moves the exact probabilities that a
    non-optimal opponent realises."""
    m = json.loads(RECORD.read_text())["results"]["misere"]

    # The standard-side numbers must reproduce the published Finding 9 rationals.
    assert m["(3,3)"]["standard"]["p_grandson_first_exact"] == "47/150"
    assert m["(4,4)"]["standard"]["p_grandson_first_exact"] == "142823/259200"
    assert m["(5,5)"]["standard"]["p_grandson_first_exact"] == "513274009/641520000"

    for size in ("(3,3)", "(4,4)", "(5,5)"):
        row = m[size]
        # A forced throw stays impossible and a weak throw stays possible, both
        # seats, with or without the rule -- so reverse-Oops is NOT necessary for
        # the Finding 9 theorem (it corrects that finding's explanation).
        for side in ("standard", "ablated"):
            assert row[side]["forced_throw_grandson_first"] is False, (size, side)
            assert row[side]["forced_throw_grandson_second"] is False, (size, side)
            assert row[side]["weak_throw_grandpa_first"] is True, (size, side)
            assert row[side]["weak_throw_grandpa_second"] is True, (size, side)
        assert row["verdicts_changed"] is False, size

    # (3,3) has no Oops at all, so the ablation must be a strict no-op there.
    assert m["(3,3)"]["any_change"] is False
    # Where Oops exists, the exact probabilities move -- and at (5,5) both seats
    # move, in OPPOSITE directions (thrower loses a tool; random player loses a
    # hazard). Check the signs, not just that something changed.
    from fractions import Fraction

    def delta(size, seat):
        row = m[size]
        return (Fraction(row["ablated"][f"p_grandson_{seat}_exact"])
                - Fraction(row["standard"][f"p_grandson_{seat}_exact"]))

    assert m["(4,4)"]["any_change"] is True
    assert delta("(4,4)", "first") == Fraction(-1, 9600)
    assert delta("(4,4)", "second") == 0
    assert delta("(5,5)", "first") == Fraction(-787, 4665600)
    assert delta("(5,5)", "second") == Fraction(5389, 8019000)
    assert delta("(5,5)", "first") < 0 < delta("(5,5)", "second")
