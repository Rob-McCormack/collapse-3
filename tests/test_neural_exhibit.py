"""Guard for Finding 16 (the neural exhibit, docs/NEURAL_EXHIBIT.md).

This is a *record* guard, not a live recompute. The exhibit is a torch-dependent
experiment quarantined in ``probes/`` whose exact decimals drift with the torch
build, so we deliberately do **not** pin them. What we lock are the claims the
README / FINDINGS / NEURAL_EXHIBIT headline actually rests on:

  * deterministic structure (decision-state census, seeded-split critical counts)
    -- these are torch-independent and must hold exactly;
  * the qualitative outcome (every certification is a forced loss);
  * the generalization thresholds (held-out optimal well above chance),
    checked as inequalities, not equalities.

If a re-run ever commits a results file that breaks the headline (a seat comes
back drawn, or held-out accuracy collapses), this fails and forces the docs to
be updated rather than silently diverging. Torch is not required to run it.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
BR = RESULTS / "neural_best_response_latest.json"
MATRIX = RESULTS / "matrix.json"

# Deterministic anchors (torch-independent: game structure + seeded splits).
DECISION_STATES = 477960          # non-terminal (4,4) states with a legal move
HELDOUT_N = 95592                 # 20% seed-fixed held-out split
CRITICAL_N = 38898                # seed-314159 held-out critical states
REMOVAL_ONLY_CRITICAL_N = 3254    # of those, removal-only-optimal

# Qualitative thresholds (neural, so inequalities only).
MIN_OPTIMAL = 0.95
MIN_CRITICAL_OPTIMAL = 0.90


def _load(path):
    assert path.exists(), f"missing committed record: {path.name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_best_response_record_backs_the_headline():
    rec = _load(BR)
    setup = rec["setup"]
    assert setup["reserves"] == [4, 4]
    assert setup["decision_states"] == DECISION_STATES

    h = rec["heldout"]
    assert h["n"] == HELDOUT_N
    assert h["critical_n"] == CRITICAL_N
    assert h["removal_only_critical_n"] == REMOVAL_ONLY_CRITICAL_N
    assert h["other_critical_n"] == CRITICAL_N - REMOVAL_ONLY_CRITICAL_N
    # Generalization: near-optimal on unseen states (threshold, not pinned).
    assert h["optimal_rate"] > MIN_OPTIMAL
    assert h["critical_optimal_rate"] > MIN_CRITICAL_OPTIMAL

    # Certification: both seats a forced loss, in a shallow, positive-integer line.
    seats = {c["seat"]: c for c in rec["best_response"]}
    assert set(seats) == {0, 1}
    for c in seats.values():
        assert c["worst_wdl"] == -1
        assert c["worst_case"] == "forced_loss"
        assert isinstance(c["depth_to_worst"], int) and 1 <= c["depth_to_worst"] <= 14
        assert c["grading"]["first_policy_positive_regret_ply"] >= 1


def test_reconciliation_matrix_is_12_of_12_forced_loss():
    rows = _load(MATRIX)
    # Two architectures (A/B) x three seeds = six nets.
    assert len(rows) == 6
    assert {(r["seed"], r["setup"]) for r in rows} == {
        (s, a) for s in (314159, 0, 1) for a in ("A", "B")
    }

    forced = 0
    for r in rows:
        counts = r["audit"]["counts"]
        assert counts["n"] == HELDOUT_N
        # remWDL / remRAW pick nearly the same removal-only set (differ by a few).
        assert abs(counts["remWDL"] - counts["remRAW"]) <= 8
        # Every net generalizes above threshold on held-out states...
        assert r["audit"]["overall"][0] > MIN_OPTIMAL
        assert r["audit"]["critical"][0] > MIN_CRITICAL_OPTIMAL
        # ...and every certification, both seats, is a forced loss.
        for seat in ("0", "1"):
            cert = r["cert"][seat]
            assert cert["worst"] == -1, (r["seed"], r["setup"], seat)
            forced += 1
    assert forced == 12  # 6 nets x 2 seats
