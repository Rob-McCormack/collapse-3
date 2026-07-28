"""Guard for the tremble-budget experiment (experiments/tremble_budget.py).

Three layers, following tests/test_evaluation_equivalence.py:

  * LIVE reproduce-or-abort at (3,3): the one-blunder set equalities, the
    independent 0-1 BFS reconstruction, BOTH halves of the mechanism gate, the
    class-ladder second blunder and what it is actually made of, and the V1/V2
    decomposition. If the engine, the canonical policy, or the budget accounting
    drifts, this fails loudly.
  * the same battery at (4,4) behind COLLAPSE3_SLOW=1 (~2.5 min), including the
    certification cliff -- the cell the whole scale-mismatch story rests on.
  * a RECORD guard on the committed results JSON so a re-run cannot silently
    change the headline.

The (3,3) tier also cross-checks this module's policy-parameterised
`compatible_extreme` against the committed one in `evaluation_equivalence`,
value and depth, so the generalization cannot quietly become a fork.

Absolute counts here are reference-policy-dependent; they are pinned because
they are the fingerprint of a specific frozen policy on a specific engine, not
because any integer is a property of the game.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

import experiments.evaluation_equivalence as ee
import experiments.tremble_budget as tb
from collapse3.enumeration import solve_all
from collapse3.game import empty_state

SLOW = os.environ.get("COLLAPSE3_SLOW") == "1"
LATEST = Path(__file__).resolve().parent.parent / "results" / "tremble_budget_latest.json"

slow_only = pytest.mark.skipif(
    not SLOW, reason="set COLLAPSE3_SLOW=1 to re-solve the (4,4) budget cells (~2.5 min)")


@lru_cache(maxsize=None)
def memo(r: int):
    return solve_all(empty_state(r, r, 0))


@lru_cache(maxsize=None)
def cell(r: int, seat: int, grading: str, policy: str):
    return tb.budget_cell(memo(r), r, seat, grading, policy)


@lru_cache(maxsize=None)
def scan(r: int, seat: int, grading: str, policy: str):
    # Same limits the experiment uses, so the gates and the record agree:
    # exhaustive at (3,3), deterministically sampled above it.
    depth, cap = (None, None) if r <= 3 else (8, 150)
    return tb.conjecture_scan(memo(r), r, seat, grading, depth, cap, policy)


def row_for(sc, root_value: int):
    return next(row for row in sc["rows"]
                if row["root_value_opponent_seat"] == root_value)


def pinned(c, budget: int) -> int:
    return next(row["pinned"] for row in c["budgets"] if row["budget"] == budget)


def worst(c, budget: int) -> str:
    return next(row["worst"] for row in c["budgets"] if row["budget"] == budget)


# ============================================ (3,3) live: agreement with Gate C

def test_support0_and_all_legal_agree_with_gate_c():
    # The cheapest available validation: this experiment's endpoints must be the
    # SAME SETS Finding 17 already committed. support(0) == optimal-only is a
    # wiring check (budget 0 is the optimal-only opponent by construction); the
    # all-legal equality is the one that constrains anything.
    for seat, opt, alle in ((0, 41, 96), (1, 367, 538)):
        agree = tb.gate_c_agreement(memo(3), 3, seat)
        assert agree["support0_equals_gate_c_optimal"] is True, seat
        assert agree["support_all_equals_gate_c_all"] is True, seat
        assert (agree["gate_c_optimal_pinned"], agree["gate_c_all_pinned"]) == (opt, alle)
        assert ee.GATE0_ANCHORS[(3, 3)][seat]["optimal"][0] == opt
        assert ee.GATE0_ANCHORS[(3, 3)][seat]["all"][0] == alle


def test_generalized_solver_matches_evaluation_equivalence():
    # Policy-parameterised compatible_extreme must reproduce ee's hardwired one
    # exactly at V1 -- value AND shortest-certificate depth, both extremes.
    for seat in (0, 1):
        assert tb.check_generalization_matches_ee(memo(3), 3, seat)


# ================================== (3,3) live: one blunder is enough (WDL/V1)

def test_one_blunder_reaches_the_whole_all_legal_support():
    # SET equality, not equal cardinality: budgets 1 and 2 expose exactly the
    # decisions an unrestricted opponent exposes, so the budget saturates at 1.
    for seat, s0, s_all in ((0, 41, 96), (1, 367, 538)):
        c = cell(3, seat, tb.GRADE_WDL, tb.POLICY_V1)
        assert c["all_legal_pinned"] == s_all
        assert pinned(c, 0) == s0
        assert pinned(c, 1) == pinned(c, 2) == s_all
        assert [row["equals_all_legal"] for row in c["budgets"]] == [False, True, True]


def test_certification_flips_at_one_blunder():
    # (3,3) seat 0 is Finding 17's failing cell: a perfect opponent leaves a
    # certified forced loss, and a single permitted mistake removes it.
    c0 = cell(3, 0, tb.GRADE_WDL, tb.POLICY_V1)
    assert (worst(c0, 0), worst(c0, 1), worst(c0, 2)) == ("LOSS", "draw", "draw")
    assert c0["certification_flips_at"] == 1
    # Seat 1 at (3,3) never had a residual to remove; it certifies at m=0.
    c1 = cell(3, 1, tb.GRADE_WDL, tb.POLICY_V1)
    assert (worst(c1, 0), worst(c1, 1)) == ("draw", "draw")
    assert c1["certification_flips_at"] == 0


def test_independent_bfs_reconstructs_every_support():
    # A distance labelling computed by a separate 0-1 BFS must rebuild each
    # support exactly: {s : min_blunders(s) <= m} == support(m).
    for seat, hist in ((0, {"0": 41, "1": 55}), (1, {"0": 367, "1": 171})):
        c = cell(3, seat, tb.GRADE_WDL, tb.POLICY_V1)
        assert c["min_blunder_histogram"] == hist
        assert c["max_min_blunders"] == 1
        assert all(c["bfs_reconstructs_support"].values())


def test_one_blunder_states_are_shallow():
    # The decisions the tremble buys are near the root, so a real evaluation
    # would reach them: nothing here depends on deep forcing lines.
    d0 = cell(3, 0, tb.GRADE_WDL, tb.POLICY_V1)["ply_depths"]["1"]
    d1 = cell(3, 1, tb.GRADE_WDL, tb.POLICY_V1)["ply_depths"]["1"]
    assert (d0["min"], d0["median"], d0["max"]) == (2, 4, 6)
    assert (d1["min"], d1["median"], d1["max"]) == (5, 5, 5)


# =============================================== (3,3) live: the mechanism gate

def test_mechanism_gate_both_halves():
    # (i) WDL: below a blunder every opponent move is zero-regret -- once the
    # game is thrown there is nothing left to throw, which is WHY one unit of
    # budget suffices. Asserting only this half would check nothing, since it
    # passes for any grading whose zero-regret sets happen to be total, so:
    # (ii) class grading must FAIL, and fail exactly at attrition-lost (-10)
    # intermediates, where a rung of the ladder is still left to fall off.
    w = tb.mechanism_gate(memo(3), 3, 0, ee.zero_regret_moves, ee.canonical_optimal)
    assert (w["below_blunder"], w["violations"]) == (2, 0)
    assert tb.mechanism_gate(memo(3), 3, 1)["violations"] == 0

    c = tb.mechanism_gate(memo(3), 3, 0, tb.zero_regret_class, ee.canonical_optimal)
    assert c["below_blunder"] == 2
    assert c["violations"] == 1
    assert c["violations_are_exactly_attrition_lost"] is True
    assert c["attrition_constrained"] == 1


# ========================================= (3,3) live: the class-ladder artifact

def test_class_grading_demands_a_second_blunder():
    c = cell(3, 0, tb.GRADE_CLASS, tb.POLICY_V1)
    assert c["min_blunder_histogram"] == {"0": 41, "1": 47, "2": 8}
    assert c["max_min_blunders"] == 2
    assert [pinned(c, m) for m in (0, 1, 2)] == [41, 88, 96]
    assert all(c["bfs_reconstructs_support"].values())
    # Two blunders are REACHABLE here but do not gate certification at (3,3):
    # the flip is still at one. Only (4,4) seat 0 turns this into a cliff.
    assert c["certification_flips_at"] == 1
    # Seat 1 never needs the second unit at either grading.
    assert cell(3, 1, tb.GRADE_CLASS, tb.POLICY_V1)["max_min_blunders"] == 1


def test_the_eight_are_one_decision_node_not_eight_facts():
    fam = tb.deepest_budget_family(memo(3), 3, 0)
    assert fam["budget"] == 2
    assert fam["states_unfolded"] == 8
    # 8 == |D4|, which invites reading the family as one position counted eight
    # times over on the un-folded state space. It is NOT: eight distinct
    # canonical keys. The coincidence is pinned here so nobody re-derives it.
    assert fam["distinct_up_to_symmetry"] == 8
    assert fam["is_one_d4_orbit"] is False
    # What they actually are: every non-zero-regret continuation of a SINGLE
    # opponent decision -- the same lone class-constrained state the mechanism
    # gate counts. The 8 is a branching factor, not eight independent findings.
    assert fam["decision_nodes"] == 1
    assert fam["family_is_exactly_the_blundering_children"] is True
    node = fam["nodes"][0]
    assert (node["legal_moves"], node["zero_regret_moves"]) == (9, 1)
    assert node["value_opponent_seat"] == -10
    assert node["min_blunders_to_node"] == 1
    # A placement-only line: reserves (1,1), no removals, four plies deep.
    assert fam["reserves"] == [(1, 1)] and fam["removals_played"] == [0]
    assert fam["ply_depths"] == [4]
    assert fam["value_trajectory_opponent_seat"] == [0, 0, -10, -10, -100]


def test_v2_decomposition_at_3_3():
    # Scale-matching the reference does NOT dissolve the second blunder here --
    # the 8 survive -- but they never gated certification at (3,3) anyway.
    v2 = cell(3, 0, tb.GRADE_CLASS, tb.POLICY_V2)
    assert v2["all_legal_pinned"] == 96
    assert v2["min_blunder_histogram"] == {"0": 41, "1": 47, "2": 8}
    assert v2["max_min_blunders"] == 2
    assert [pinned(v2, m) for m in (0, 1, 2)] == [41, 88, 96]
    assert v2["certification_flips_at"] == 1
    m = tb.mechanism_gate(memo(3), 3, 0, tb.zero_regret_class, tb.canonical_optimal_v2)
    assert m["attrition_constrained"] == 1


# ============================================ (3,3) live: policy dependence

def test_reversed_tiebreak_moves_the_counts_but_not_the_pattern():
    # The robustness control, and the direct demonstration behind this module's
    # standing caveat: swap the tie-break and every absolute count moves, while
    # the flip-at-one-blunder pattern is untouched.
    r0 = cell(3, 0, tb.GRADE_WDL, tb.POLICY_V1R)
    v0 = cell(3, 0, tb.GRADE_WDL, tb.POLICY_V1)
    assert (pinned(r0, 0), r0["all_legal_pinned"]) == (49, 192)
    assert (pinned(v0, 0), v0["all_legal_pinned"]) == (41, 96)
    assert r0["max_min_blunders"] == 1
    assert (worst(r0, 0), worst(r0, 1)) == ("LOSS", "draw")
    assert r0["certification_flips_at"] == 1


# ================================ (3,3) live: the conjecture's own falsification

def test_conjecture_scan_is_exhaustive_at_3_3():
    # Not a sample: every non-terminal candidate-to-move state is used as a
    # root, so a violation at (3,3) could not hide behind sampling.
    for seat, roots in ((0, 10882), (1, 15228)):
        for grading, policy in ((tb.GRADE_WDL, tb.POLICY_V1),
                                (tb.GRADE_CLASS, tb.POLICY_V1),
                                (tb.GRADE_CLASS, tb.POLICY_V2)):
            sc = scan(3, seat, grading, policy)
            assert sc["exhaustive"] is True
            assert sc["roots_tested"] == roots
            # (3,3) is clean even for the mismatched pair. That is exactly why
            # the (4,4) gate below exists: the rule looks fine on the small
            # board, as in Finding 8.
            assert sc["bound_holds_everywhere"] is True, (seat, grading, policy)
            assert sc["budgets_above_two_observed"] is False


def test_the_equality_clause_is_refuted_at_3_3():
    # Where the class bound permits three, one is what actually happens. This
    # is why the recorded conjecture carries no equality condition.
    sc = scan(3, 1, tb.GRADE_CLASS, tb.POLICY_V1)
    row = row_for(sc, 10)
    assert row["classes_below"] == 3
    assert row["roots_tested"] == 882
    assert row["max_budget_observed"] == 1
    assert row["attains_bound"] is False
    # The bound is attained only where it is small; it never binds above 2.
    assert 10 not in sc["bound_attained_somewhere"]


def test_scale_matching_is_what_the_bound_depends_on():
    assert tb.is_scale_matched(tb.GRADE_WDL, tb.POLICY_V1) is True
    assert tb.is_scale_matched(tb.GRADE_CLASS, tb.POLICY_V2) is True
    # The one combination where the reference optimizes a coarser scale than
    # the grader scores -- and the only one that ever breaks the bound.
    assert tb.is_scale_matched(tb.GRADE_CLASS, tb.POLICY_V1) is False
    assert tb.classes_below(0, tb.GRADE_CLASS) == 2
    assert tb.classes_below(0, tb.GRADE_WDL) == 1
    assert tb.classes_below(-100, tb.GRADE_CLASS) == 0
    assert tb.classes_below(10, tb.GRADE_CLASS) == 3
    assert tb.classes_below(100, tb.GRADE_CLASS) == 4


# =================================================== (4,4) behind the slow flag

@slow_only
def test_one_blunder_is_enough_at_4_4():
    for seat, s0, s_all, hist in (
            (0, 10, 447, {"0": 10, "1": 437}),
            (1, 1241, 3982, {"0": 1241, "1": 2741})):
        c = cell(4, seat, tb.GRADE_WDL, tb.POLICY_V1)
        assert c["all_legal_pinned"] == s_all
        assert pinned(c, 0) == s0
        assert pinned(c, 1) == pinned(c, 2) == s_all
        assert [row["equals_all_legal"] for row in c["budgets"]] == [False, True, True]
        assert c["min_blunder_histogram"] == hist
        assert c["max_min_blunders"] == 1
        assert all(c["bfs_reconstructs_support"].values())
        # Both seats fail at (4,4) under a perfect opponent, and one mistake
        # fixes both -- the effect is two-sided where Finding 17 says it is.
        assert (worst(c, 0), worst(c, 1)) == ("LOSS", "draw")
        assert c["certification_flips_at"] == 1
    assert tb.gate_c_agreement(memo(4), 4, 0)["support_all_equals_gate_c_all"] is True


@slow_only
def test_mechanism_gate_both_halves_at_4_4():
    for seat, below in ((0, 49), (1, 221)):
        w = tb.mechanism_gate(memo(4), 4, seat)
        assert (w["below_blunder"], w["violations"]) == (below, 0)
    c = tb.mechanism_gate(memo(4), 4, 0, tb.zero_regret_class, ee.canonical_optimal)
    assert c["violations"] == 1
    assert c["violations_are_exactly_attrition_lost"] is True


@slow_only
def test_the_certification_cliff_at_4_4_seat_0():
    # THE cliff cell, and the only one anywhere in this experiment: under class
    # grading a single permitted mistake is not enough, m=1 still leaves a
    # certified forced loss, and certification waits for m=2.
    c = cell(4, 0, tb.GRADE_CLASS, tb.POLICY_V1)
    assert c["min_blunder_histogram"] == {"0": 10, "1": 429, "2": 8}
    assert c["max_min_blunders"] == 2
    assert [pinned(c, m) for m in (0, 1, 2)] == [10, 439, 447]
    assert [worst(c, m) for m in (0, 1, 2)] == ["LOSS", "LOSS", "draw"]
    assert c["certification_flips_at"] == 2
    assert cell(4, 1, tb.GRADE_CLASS, tb.POLICY_V1)["max_min_blunders"] == 1


@slow_only
def test_the_cliff_is_scale_mismatch_not_ladder_depth():
    # Scale-match the reference to the grader and the cliff is gone: the eight
    # states are unreachable, the budget drops back to one, m=1 exposes the
    # whole support and certifies. The ladder did not get shallower -- the
    # candidate stopped being indifferent to rungs the grader was scoring.
    v2 = cell(4, 0, tb.GRADE_CLASS, tb.POLICY_V2)
    assert v2["all_legal_pinned"] == 438
    assert v2["min_blunder_histogram"] == {"0": 10, "1": 428}
    assert v2["max_min_blunders"] == 1
    assert pinned(v2, 1) == 438
    assert v2["budgets"][1]["equals_all_legal"] is True
    assert v2["certification_flips_at"] == 1
    m = tb.mechanism_gate(memo(4), 4, 0, tb.zero_regret_class, tb.canonical_optimal_v2)
    assert (m["violations"], m["attrition_constrained"]) == (0, 0)


@slow_only
def test_the_eight_at_4_4_are_also_one_decision_node():
    fam = tb.deepest_budget_family(memo(4), 4, 0)
    assert fam["states_unfolded"] == 8
    assert fam["distinct_up_to_symmetry"] == 8 and fam["is_one_d4_orbit"] is False
    assert fam["decision_nodes"] == 1
    assert fam["family_is_exactly_the_blundering_children"] is True
    assert fam["reserves"] == [(1, 1)] and fam["removals_played"] == [0]
    assert fam["ply_depths"] == [6]
    # The -100 -> -10 step is the reference handing a rung back: V1 is WDL-
    # optimal and therefore indifferent between losing by a line and losing on
    # the tiebreaker, which is exactly the indifference V2 removes.
    assert fam["value_trajectory_opponent_seat"] == [0, 0, -100, -100, -100, -10, -100]


@slow_only
def test_reversed_tiebreak_at_4_4():
    for seat, s0, s_all in ((0, 12, 1272), (1, 966, 3568)):
        c = cell(4, seat, tb.GRADE_WDL, tb.POLICY_V1R)
        assert (pinned(c, 0), c["all_legal_pinned"]) == (s0, s_all)
        assert c["max_min_blunders"] == 1
        assert (worst(c, 0), worst(c, 1)) == ("LOSS", "draw")


@slow_only
def test_the_class_count_bound_fails_under_the_mismatched_pair_at_4_4():
    # The refutation. A class-graded opponent against the WDL-optimal V1
    # reference exceeds the bound in three separate root classes, including
    # roots it has ALREADY lost outright, where zero further falls should be
    # possible. The candidate's indifference between -100 and -10 hands the rung
    # back and the ladder gets re-descended -- the same indifference behind the
    # seat-0 certification cliff.
    sc = scan(4, 0, tb.GRADE_CLASS, tb.POLICY_V1)
    assert sc["scale_matched"] is False
    assert sc["bound_holds_everywhere"] is False
    assert sc["budgets_above_two_observed"] is True
    violations = {row["root_value_opponent_seat"]:
                  (row["classes_below"], row["max_budget_observed"])
                  for row in sc["rows"] if row["exceeds_bound"]}
    assert violations == {0: (2, 3), -10: (1, 2), -100: (0, 2)}


@slow_only
def test_the_bound_survives_under_every_scale_matched_pair_at_4_4():
    for seat in (0, 1):
        for grading, policy in ((tb.GRADE_WDL, tb.POLICY_V1),
                                (tb.GRADE_CLASS, tb.POLICY_V2)):
            sc = scan(4, seat, grading, policy)
            assert sc["scale_matched"] is True
            assert sc["bound_holds_everywhere"] is True, (seat, grading, policy)
    # Seat 1 is clean even mismatched: the failure is one seat on one board.
    assert scan(4, 1, tb.GRADE_CLASS, tb.POLICY_V1)["bound_holds_everywhere"] is True


# ================================================================ record guard

def test_committed_record_backs_the_headline():
    assert LATEST.exists(), "run: python -m experiments.tremble_budget 3 --slow"
    rec = json.loads(LATEST.read_text(encoding="utf-8"))
    assert rec["experiment"] == "tremble_budget"
    res = rec["results"]

    # The standing caveat must travel with the numbers.
    assert "reference-policy-dependent" in res["note"]
    conj = res["conjecture"]
    assert "REFUTED" in conj["status"]
    assert "at most the number of value classes" in conj["statement"]
    assert conj["refutations"] and conj["what_survives"] and conj["why_not_a_law"]

    # The record must carry the falsification attempt, not just the claim.
    sc = res["sizes"]["(3,3)"]["seats"]["0"]["conjecture_scan"]
    assert sc["wdl_v1"]["exhaustive"] is True
    assert sc["class5_v1"]["scale_matched"] is False
    assert sc["class5_v2"]["scale_matched"] is True

    s0 = res["sizes"]["(3,3)"]["seats"]["0"]
    wdl = s0["wdl_v1"]
    assert wdl["all_legal_pinned"] == 96
    assert [row["pinned"] for row in wdl["budgets"]] == [41, 96, 96]
    assert [row["worst"] for row in wdl["budgets"]] == ["LOSS", "draw", "draw"]
    assert wdl["max_min_blunders"] == 1
    assert s0["class_v1"]["max_min_blunders"] == 2
    assert s0["two_blunder_family"]["decision_nodes"] == 1
    assert s0["two_blunder_family"]["is_one_d4_orbit"] is False

    # If the record carried (4,4), lock the cliff, its dissolution, and the
    # refutation of the unconditional class-count bound.
    if "(4,4)" in res["sizes"]:
        f0 = res["sizes"]["(4,4)"]["seats"]["0"]
        assert [row["worst"] for row in f0["wdl_v1"]["budgets"]] == ["LOSS", "draw", "draw"]
        assert [row["worst"] for row in f0["class_v1"]["budgets"]] == ["LOSS", "LOSS", "draw"]
        assert f0["class_v2"]["max_min_blunders"] == 1
        assert f0["class_v2"]["all_legal_pinned"] == 438
        f_sc = f0["conjecture_scan"]
        assert f_sc["class5_v1"]["bound_holds_everywhere"] is False
        assert f_sc["wdl_v1"]["bound_holds_everywhere"] is True
        assert f_sc["class5_v2"]["bound_holds_everywhere"] is True

    # If the record carried (5,5), lock the support half of the headline. There
    # is no certification there: the pinned solves are memory-bound at 12.7M
    # states, the same wall that kept Finding 17's Gate C at (4,4).
    five = res.get("headline_5_5")
    if five:
        assert five["certified"] is False
        for seat in ("0", "1"):
            c = five["seats"][seat]["wdl_v1"]
            assert c["max_min_blunders"] == 1
            assert c["budgets"][1]["equals_all_legal"] is True
            assert all(c["bfs_reconstructs_support"].values())
        # Cross-experiment agreement: support(0) at (5,5) must be the optimal-only
        # support Finding 17's Gate A already pinned there, reached by a
        # different traversal in a different module.
        ee_rec = json.loads((LATEST.parent / "evaluation_equivalence_latest.json")
                            .read_text(encoding="utf-8"))
        gate_a5 = ee_rec["results"].get("gate_a", {}).get("(5,5)")
        if gate_a5:
            for seat in ("0", "1"):
                assert (five["seats"][seat]["gate_c_agreement"]["gate_c_optimal_pinned"]
                        == gate_a5[seat]["pinned"]), seat
