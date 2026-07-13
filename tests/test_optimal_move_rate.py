"""Guard the README/Finding 1 headline pair against the shipped results JSON.

The marquee stat -- "a one-ply agent posts its highest optimal-move rate (78%)
against the opponent it loses to 84% of the time (and wins 0%)" -- must trace
to results/optimal_move_rate_latest.json, and the same numbers must agree with
the random-baseline row of the reworked depth sweep (Finding 8 proved the
bead-count one-ply agent is random-equivalent on critical decisions, so the
two independent runs land on identical values; if either file is regenerated
and drifts, this test flags the headline as citing a superseded table).
"""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_recorded_headline_pair_78_and_84():
    rows = json.loads((RESULTS / "optimal_move_rate_latest.json").read_text())["results"]["rows"]
    by_opp = {r["opponent"]: r for r in rows}
    vs_opt = by_opp["optimal"]
    # ...the 78% aggregate is the HIGHEST across all opponents tested...
    assert vs_opt["optimal_rate_overall"] == max(r["optimal_rate_overall"] for r in rows)
    assert vs_opt["optimal_rate_overall"] == pytest.approx(0.783, abs=0.001)
    # ...against the opponent it loses to 84% of the time and never beats.
    assert vs_opt["loss_rate"] == pytest.approx(0.84, abs=0.001)
    assert vs_opt["win_rate"] == 0.0
    # The critical-only rate is the mirage's other half (0.359, ~random).
    assert vs_opt["optimal_rate_critical"] == pytest.approx(0.359, abs=0.001)


def test_headline_numbers_agree_with_depth_sweep_random_row():
    vs_opt = {r["opponent"]: r for r in json.loads(
        (RESULTS / "optimal_move_rate_latest.json").read_text())["results"]["rows"]}["optimal"]
    rand = {r["policy"]: r for r in json.loads(
        (RESULTS / "depth_sweep_latest.json").read_text())["results"]["rows"]
        if r["opponent"] == "optimal"}["rand"]
    for key in ("optimal_rate_overall", "optimal_rate_critical", "loss_rate", "win_rate"):
        assert vs_opt[key] == pytest.approx(rand[key], abs=1e-9), key
