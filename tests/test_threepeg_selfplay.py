"""Self-play on the Three-Peg sibling: guard the exact, seeded results.

Self-play training is stochastic, but the seeds are fixed, so every number is
deterministic and reproducible. The default suite runs only cheap guards on the
shipped results JSON plus the qualitative claims of the finding; the heavy
recompute (multi-seed training + best-response) is gated behind COLLAPSE3_SLOW.

The three claims guarded (Finding 14):
  * global blindness: on-policy regret << uniform (global) regret, and coverage
    is well below 1 -- self-play looks perfect where it plays, mediocre elsewhere;
  * seat-0 (the winnable seat) robustness / exploitability as recorded;
  * transfer across the boundary: a reserve-blind policy trained at (5,5) opens
    on the centre and is forced to lose at (7,7) (centre inverts, Finding 12/13).
"""

import json
import os
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results" / "threepeg_selfplay_latest.json"


@pytest.fixture(scope="module")
def record():
    return json.loads(RESULTS.read_text())


def test_provenance_and_scope(record):
    res = record["results"]
    assert res["placement_pegs"] == [0, 1, 2]
    assert res["seeds"] == [0, 1, 2, 3, 4]
    assert res["variant"] == "Three-Peg Collapse (self-play)"


def test_global_blindness(record):
    """Coverage is partial and falls with size; and for FULL observation (no
    aliasing floor, so uniform regret is a pure coverage/learning gap) on-policy
    regret stays far below global regret, with the gap widening as the game grows.
    """
    by = record["results"]["by_size"]
    cov = [by[f"{r}_{r}"]["regimes"]["full"]["coverage_mean"] for r in (4, 5, 6, 7)]
    assert all(c < 1.0 for c in cov)          # never full coverage
    assert cov[0] > cov[-1]                    # coverage drops with size

    reps = [by[f"{r}_{r}"]["regimes"]["full"] for r in (4, 5, 6, 7)]
    for rep in reps:
        assert rep["onpolicy_self_mean"] < rep["uniform_regret_mean"]
    gap_small = reps[0]["uniform_regret_mean"] - reps[0]["onpolicy_self_mean"]
    gap_big = reps[-1]["uniform_regret_mean"] - reps[-1]["onpolicy_self_mean"]
    assert gap_big > gap_small                 # on-policy/global gap widens


def test_global_regret_grows_with_size(record):
    """Uniform (global) regret at (7,7) exceeds (4,4) -- blindness worsens."""
    by = record["results"]["by_size"]
    for regime in ("full", "hide_reserves"):
        assert by["7_7"]["regimes"][regime]["uniform_regret_mean"] > \
            by["4_4"]["regimes"][regime]["uniform_regret_mean"]


def test_transfer_across_boundary(record):
    """Reserve-blind self-play trained at (5,5) carries the centre opening into
    (7,7), where the centre loses -- every seed forced to lose."""
    tr = record["results"]["transfer"]
    assert tr["trained_at"] == "5_5" and tr["regime"] == "hide_reserves"
    assert tr["seeds_open_centre"] == 5
    assert tr["seeds_forced_loss_at_7_7"] == 5
    for row in tr["per_seed"]:
        assert row["opening_peg"] == 1                       # centre
        assert row["opening_value_5_5"] == "win"            # optimal at (5,5)
        assert row["opening_value_7_7"] == "loss"           # inverts at (7,7)
        assert row["best_response_7_7_seat0"] == "loss"     # exactly forced


def test_seat1_is_game_value_not_exploit(record):
    """Seat 1 is theoretically lost at every size (root is a P0 win), so its
    best-response is always a loss -- confirming the value, not an exploit."""
    by = record["results"]["by_size"]
    for info in by.values():
        for rep in info["regimes"].values():
            assert all(w == -1 for w in rep["per_seed"]["best_response_seat1"])


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to recompute self-play (multi-seed training, ~minutes)",
)
def test_recompute_transfer_matches_record():
    from collapse3.enumeration import solve_all
    from collapse3.game import empty_state, placement_pegs
    from experiments.threepeg_selfplay import transfer_across_boundary
    from experiments.threepeg_floor import PLACEMENT_ROW

    with placement_pegs(PLACEMENT_ROW):
        memos = {r: solve_all(empty_state(r, r)) for r in (5, 6, 7)}
        tr = transfer_across_boundary(memos, episodes=50_000)
    assert tr["seeds_open_centre"] == 5
    assert tr["seeds_forced_loss_at_7_7"] == 5
