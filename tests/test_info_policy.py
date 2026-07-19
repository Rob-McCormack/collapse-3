"""Exact value-of-information / adaptive-query floors (info_policy).

Reproduce-or-abort: the light (3,3) census is recomputed live and locked to the
digit; the heavy (4,4) census is guarded from the recorded results file. The
marquee claim is the *sparse reserve VOI*: on the mask-aware interface a tiny
fraction of decisions carries the entire irreducible reserve floor.

Cross-check: at (4,4) the reserve axis'  beneficial-query groups must equal the
masked-floor conflict groups (1256) -- the states where reserves help are
exactly the observation groups with an optimal-action conflict.
"""

import json
from pathlib import Path

from collapse3.enumeration import solve_all
from collapse3.game import empty_state
from experiments.info_policy import AXES, decisions_of, info_policy

RESULTS = Path(__file__).resolve().parent.parent / "results"
INFO = RESULTS / "info_policy_latest.json"
MASKED = RESULTS / "masked_floor_latest.json"


def _live(size):
    decisions = decisions_of(solve_all(empty_state(size, size)))
    return {name: info_policy(decisions, base, refine)
            for name, (base, refine) in AXES.items()}


def test_3_3_recomputes_to_the_digit():
    axes = _live(3)
    m, r = axes["mask"], axes["reserve"]

    # Mask axis: 0.827% of states buy back the whole mask-blind floor.
    assert m["decision_states"] == 26110
    assert m["base_floor"] == 0.0027575641516660284
    assert m["refined_floor"] == 0.0
    assert m["beneficial_query_groups"] == 72
    assert m["beneficial_query_states"] == 216
    assert abs(m["beneficial_query_state_fraction"] - 0.008272692454998084) < 1e-15

    # Reserve axis: missing != useful. Reserves hidden, but zero decision value.
    assert r["base_floor"] == 0.0
    assert r["refined_floor"] == 0.0
    assert r["beneficial_query_groups"] == 0
    assert r["beneficial_query_states"] == 0


def test_recorded_4_4_sparse_reserve_voi():
    by_size = json.loads(INFO.read_text())["results"]["by_size"]
    r = by_size["4_4"]["axes"]["reserve"]
    m = by_size["4_4"]["axes"]["mask"]

    # The load-bearing result: ~1.07% of decisions carry the entire reserve floor.
    assert r["decision_states"] == 477960
    assert abs(r["base_floor"] - 0.0026278349652690603) < 1e-15
    assert r["refined_floor"] == 0.0
    assert r["beneficial_query_groups"] == 1256
    assert r["beneficial_query_states"] == 5120
    assert abs(r["beneficial_query_state_fraction"] - 0.010712193488994895) < 1e-12

    # Mask axis is diffuse, not sparse (13%), and querying it everywhere only
    # reduces to the mask-aware interface -- its residual IS the reserve floor.
    assert m["beneficial_query_state_fraction"] > 0.13
    assert abs(m["refined_floor"] - r["base_floor"]) < 1e-15


def test_reserve_voi_states_are_the_masked_conflict_groups():
    """The states that need reserves == the masked-floor optimal-action conflicts."""
    info = json.loads(INFO.read_text())["results"]["by_size"]["4_4"]["axes"]["reserve"]
    masked = json.loads(MASKED.read_text())["results"]["by_size"]["4_4"]["rows"]
    assert info["beneficial_query_groups"] == masked["hide_reserves"]["masked_conflict_groups"]


def test_cost_ladder_is_monotone_and_consistent():
    by_size = json.loads(INFO.read_text())["results"]["by_size"]
    for size in by_size.values():
        for axis in size["axes"].values():
            rows = axis["rows"]
            # refined floor is a true lower bound of the base floor
            assert axis["refined_floor"] <= axis["base_floor"] + 1e-15
            # zero-cost residual == refined floor; querying is fully worthwhile
            assert abs(rows[0]["residual_regret"] - axis["refined_floor"]) < 1e-15
            # as query cost rises, you query no more and residual regret no less
            fracs = [row["query_state_fraction"] for row in rows]
            resid = [row["residual_regret"] for row in rows]
            assert fracs == sorted(fracs, reverse=True)
            assert resid == sorted(resid)
