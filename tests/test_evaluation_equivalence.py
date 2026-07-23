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


def test_anchor_table_is_internally_consistent():
    # Guard the frozen anchor table itself against accidental edits.
    a = GATE0_ANCHORS[(3, 3)]
    assert a[0]["optimal"] == (41, "LOSS")
    assert a[0]["all"] == (96, "draw")
    assert a[0]["hash"] == "b9204bd62ed5e8cb"
    assert a[1]["hash"] == "f4f8e9548e31e5b5"


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
