"""Sandbagging / misere solves: throwing the game is provably hard (Finding).

Locks the two claims that must not drift:
  * FORCED THROW is impossible -- grandpa cannot force his own loss (a theorem,
    at the solved sizes), even at (5,5);
  * WEAK THROW is possible -- he can guarantee he never wins;
  * the random-grandson win probabilities are exact rationals (model-dependent,
    not a theorem).

Fast checks re-solve (3,3) and the (cheap, hard-pruned) (5,5) binary solves in
the default suite; (4,4) re-solves sit behind COLLAPSE3_SLOW=1. A recorded-JSON
test guards every shipped number (incl. the ~100s (5,5) expectimax) so no bundle
can ship contradicting the results file.
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.sandbagging import (
    forced_throw_forceable,
    random_grandson_prob,
    weak_throw_forceable,
)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "sandbagging_latest.json"


def test_forced_throw_is_impossible_at_3_3():
    # Grandpa cannot force his own loss from either seat.
    assert not forced_throw_forceable(3, grandson_seat=0)
    assert not forced_throw_forceable(3, grandson_seat=1)


def test_weak_throw_is_possible_at_3_3():
    # ...but he can guarantee he never wins.
    assert weak_throw_forceable(3, grandpa_seat=0)
    assert weak_throw_forceable(3, grandpa_seat=1)


def test_random_grandson_probabilities_are_exact_at_3_3():
    assert random_grandson_prob(3, grandson_seat=0) == Fraction(47, 150)
    assert random_grandson_prob(3, grandson_seat=1) == Fraction(520, 8019)


def test_forced_throw_still_impossible_and_weak_throw_possible_at_5_5():
    # The binary solves prune hard, so (5,5) is cheap even though the full solve
    # is 12.7M states -- lock the headline "not even at (5,5)" theorem directly.
    assert not forced_throw_forceable(5, grandson_seat=0)
    assert not forced_throw_forceable(5, grandson_seat=1)
    assert weak_throw_forceable(5, grandpa_seat=0)
    assert weak_throw_forceable(5, grandpa_seat=1)


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to re-solve the (4,4) expectimax (~9s)",
)
def test_4_4_reproduces():
    assert not forced_throw_forceable(4, grandson_seat=0)
    assert not forced_throw_forceable(4, grandson_seat=1)
    assert weak_throw_forceable(4, grandpa_seat=0)
    assert weak_throw_forceable(4, grandpa_seat=1)
    assert random_grandson_prob(4, grandson_seat=0) == Fraction(142823, 259200)
    assert random_grandson_prob(4, grandson_seat=1) == Fraction(53470343, 264627000)


def test_zero_regret_steering_counterexample():
    """Sandbagging inside the optimal set is invisible to a value oracle.

    At (3,3) with the thrower moving first, every opening preserves the draw
    (zero WDL regret), yet the openings differ in P(random grandson wins):
    edge 520/8019 beats centre 433/8019. A thrower picks the edge and pays
    zero regret -- the oracle audits thrown value, not thrown intent.
    (Counterexample from external review; locked here so the docs' claim
    stays calibrated.)
    """
    from collapse3.enumeration import solve_all, wdl
    from collapse3.game import apply_move, empty_state, get_legal_moves, tt_key

    memo = solve_all(empty_state(3, 3))
    root = empty_state(3, 3, 0)
    grandson_seat, grandpa = 1, 0

    tmemo = {}

    def throw_value(state):
        from experiments.sandbagging import _seat_wins, _terminal_score
        score = _terminal_score(state)
        if score is not None:
            return Fraction(1) if _seat_wins(score, grandson_seat) else Fraction(0)
        key = tt_key(state)
        if key in tmemo:
            return tmemo[key]
        moves = get_legal_moves(state)
        vals = [throw_value(apply_move(state, m)) for m in moves]
        v = max(vals) if state.turn == grandpa else sum(vals, Fraction(0)) / len(vals)
        tmemo[key] = v
        return v

    openings = get_legal_moves(root)
    # Every opening is zero-regret in the normal game (the draw is preserved).
    for m in openings:
        assert wdl(memo[apply_move(root, m)], 0) == 0
    # ...yet their throw-values differ: the oracle cannot see the choice.
    tv = {m: throw_value(apply_move(root, m)) for m in openings}
    assert tv[("place", 1)] == Fraction(520, 8019)   # edge: thrower's optimum
    assert tv[("place", 4)] == Fraction(433, 8019)   # centre: worst for throwing
    assert max(tv.values()) > min(tv.values())


def test_recorded_results_are_consistent():
    """Guard every shipped number (incl. (5,5)) against the results JSON."""
    data = json.loads(RESULTS.read_text())["results"]["by_size"]
    for size in ("3_3", "4_4", "5_5"):
        cell = data[size]
        assert cell["forced_throw"] == {"grandson_first": False, "grandson_second": False}
        assert cell["weak_throw"] == {"grandpa_first": True, "grandpa_second": True}
    probs = {s: data[s]["random_grandson_win_prob"] for s in ("3_3", "4_4", "5_5")}
    assert probs["3_3"]["grandson_first_exact"] == "47/150"
    assert probs["3_3"]["grandson_second_exact"] == "520/8019"
    assert probs["4_4"]["grandson_first_exact"] == "142823/259200"
    assert probs["4_4"]["grandson_second_exact"] == "53470343/264627000"
    assert probs["5_5"]["grandson_first_exact"] == "513274009/641520000"
    assert probs["5_5"]["grandson_second_exact"] == "132796217934929/326951951040000"
    # win-probability grows with size (grandpa throws better, never forces it)
    firsts = [float(Fraction(probs[s]["grandson_first_exact"])) for s in ("3_3", "4_4", "5_5")]
    assert firsts[0] < firsts[1] < firsts[2] < 1.0
