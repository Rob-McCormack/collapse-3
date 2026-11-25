"""Guard for the evaluation-equivalence experiment (experiments/evaluation_equivalence.py).

Two layers:

  * a LIVE reproduce-or-abort of the (2,2)+(3,3) Gate 0 anchors -- this locks the
    exact ply-depth invariant, the frozen canonical-policy hashes, the
    best-compatible == game-value assertion, and the headline H1 signal
    (at (3,3) seat 0 an optimal-only evaluator leaves a certified forced loss
    that an all-legal evaluator rules out). If the engine or the canonical policy
    drifts, this fails loudly.
  * a RECORD guard on the committed results_latest.json, so a re-run that changes
    the headline cannot land silently.

Torch-free and fast (a couple of seconds for the (2,2)+(3,3) solves).
"""

import json
from pathlib import Path

from experiments.evaluation_equivalence import (
    GATE0_ANCHORS,
    _gate_a,
    reproduce_gate0,
    verify_depth_formula,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
LATEST = RESULTS / "evaluation_equivalence_latest.json"


def test_depth_formula_is_exact_and_a_state_invariant():
    # The closed form plies = 2*placed - on_board matches BFS everywhere, so
    # depth is a genuine state invariant (no state reachable at two depths).
    assert verify_depth_formula(2) == (4051, 0)
    assert verify_depth_formula(3) == (97093, 0)


def test_gate0_reproduces_all_anchors():
    # Raises on any mismatch (counts, policy hash, control outcome, or the
    # optimal-only / all-legal compatible outcomes). Also exercises the
    # best-compatible == game-value assertion inside identifiability().
    record = reproduce_gate0()
    assert set(record["sizes"]) == {"(2,2)", "(3,3)"}
    # The H1 headline, pinned exactly: optimal-only certifies less than all-legal.
    s0 = record["sizes"]["(3,3)"]["seat0"]["benchmark"]
    assert s0["optimal"]["pinned"] == 41
    assert s0["optimal"]["worst"] == "LOSS"
    assert s0["optimal"]["identified"] is False
    assert s0["all"]["pinned"] == 96
    assert s0["all"]["worst"] == "draw"
    assert s0["all"]["identified"] is True


def test_gate_d_control_b_is_degenerate_and_labelled_as_such():
    # Gate D's 'in_support' control cannot test strategic-vs-random selection.
    # adversarial_support(plies) takes EVERY candidate decision at depth <= plies,
    # and the universe is the same traversal uncapped, so the strategic set is
    # exactly the universe's depth-truncation. Matching its ply-depth histogram
    # therefore demands the whole pool at every depth and the draw has no freedom:
    # the control returns the identical set. Verified at (3,3) and (4,4) in the
    # correction; (3,3) is the fast guard here.
    #
    # If a future change gives the control real freedom, this test SHOULD fail --
    # at which point H2 becomes testable and the docs must stop saying "untested".
    from collapse3.enumeration import solve_all
    from collapse3.game import empty_state
    import experiments.evaluation_equivalence as ee

    memo = solve_all(empty_state(3, 3, 0))
    for seat in (0, 1):
        expo = ee.benchmark_support(memo, 3, seat, "all")
        for plies in (2, 4, 6, 8):
            sup = ee.adversarial_support(memo, 3, seat, plies)
            for seed in (0, 1, 2):
                ctl = ee.depth_matched_random(memo, 3, seat, sup, seed, expo)
                assert ctl == sup, (
                    f"control B differs from strategic at seat {seat}, {plies} plies, "
                    f"seed {seed} -- H2 may now be testable; update the docs")


def test_h2_is_not_claimed_as_falsified_anywhere():
    # The degenerate control was once reported as falsifying H2. Guard the
    # retraction so it cannot creep back into the prose or the experiment.
    root = Path(__file__).resolve().parent.parent
    targets = [
        root / "README.md",
        root / "docs" / "EVALUATION_EQUIVALENCE.md",
        root / "docs" / "FINDINGS.md",
        root / "docs" / "BEAT_THE_BEST.md",
        root / "experiments" / "evaluation_equivalence.py",
    ]
    banned = ("H2 is falsified", "H2 falsified", "buys nothing over same-universe")
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text()
        for phrase in banned:
            assert phrase not in text, f"{path.name} still claims: {phrase!r}"


def test_anchor_table_is_internally_consistent():
    # Guard the frozen anchor table itself against accidental edits.
    a = GATE0_ANCHORS[(3, 3)]
    assert a[0]["optimal"] == (41, "LOSS")
    assert a[0]["all"] == (96, "draw")
    assert a[0]["hash"] == "b9204bd62ed5e8cb"
    assert a[1]["hash"] == "f4f8e9548e31e5b5"


def test_gate_a_minimum_mutation_is_one_at_3_3():
    # Gate A (live, ~2s): a candidate that passed the optimal-opponent evaluation
    # needs only a SINGLE non-canonical decision (at depth 6) to be force-losable
    # at (3,3) seat 0; seat 1 is not force-losable under that same support.
    a = _gate_a((3,))["(3,3)"]
    assert a["0"]["forced_losable"] is True
    assert a["0"]["min_mutations"] == 1
    assert a["0"]["min_depth_at_min_mutations"] == 6
    assert a["1"]["forced_losable"] is False
    assert a["1"]["min_mutations"] is None


def test_committed_record_backs_the_headline():
    assert LATEST.exists(), "run: python -m experiments.evaluation_equivalence 3"
    rec = json.loads(LATEST.read_text(encoding="utf-8"))
    assert rec["experiment"] == "evaluation_equivalence"
    gate0 = rec["results"]["gate0"]
    assert gate0["convention"]["policy"] == "CANONICAL_POLICY_V1"
    s0 = gate0["sizes"]["(3,3)"]["seat0"]["benchmark"]
    assert s0["optimal"]["worst"] == "LOSS" and s0["optimal"]["identified"] is False
    assert s0["all"]["worst"] == "draw" and s0["all"]["identified"] is True
    # Best-compatible is always the game value (a draw here), both seats.
    for seat in ("seat0", "seat1"):
        assert gate0["sizes"]["(3,3)"][seat]["control_all_pinned"]["best"] == "draw"

    # If the committed record carried the (4,4) Gate C headline, lock it too:
    # the optimal-only/all-legal gap is two-sided at (4,4).
    gate_c = rec["results"]["gate_c"]
    if "(4,4)" in gate_c:
        for seat in ("0", "1"):
            b = gate_c["(4,4)"][seat]["benchmark"]
            assert b["optimal"]["worst"] == "LOSS", seat
            assert b["optimal"]["identified"] is False, seat
            assert b["all"]["worst"] == "draw" and b["all"]["identified"] is True, seat

    # If the committed record carried Gate A, lock the minimum-mutation headline:
    # a single non-canonical decision suffices, and it is two-sided at (4,4).
    gate_a = rec["results"].get("gate_a")
    if gate_a and "(3,3)" in gate_a:
        s0 = gate_a["(3,3)"]["0"]
        assert s0["forced_losable"] is True and s0["min_mutations"] == 1
    for size in ("(4,4)", "(5,5)"):
        if gate_a and size in gate_a:
            for seat in ("0", "1"):
                m = gate_a[size][seat]
                assert m["forced_losable"] is True and m["min_mutations"] == 1, \
                    (size, seat)
