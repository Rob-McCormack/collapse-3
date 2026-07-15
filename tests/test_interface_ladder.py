"""The aliasing floor is a property of the interface: guard the exact ladder.

The reserve feature's cost is not one number. Under ``hide_reserves`` it is
0.0805 to a (board, cooldown) policy, ~0.0026 once the legal-move mask is folded
in (what the trained QAgent actually sees), and 0.0000 with destroyed-bead
memory. These three are exact, opponent-independent, uniform-over-decision-state
floors. This locks their values and the monotonicity mask-blind >= mask-aware >=
+memory == 0, which is the synthesis of Findings 4 and 7.
"""

import json
import os
from pathlib import Path

import pytest

from experiments.interface_ladder import ladder

RESULTS = Path(__file__).resolve().parent.parent / "results" / "interface_ladder_latest.json"


def test_ladder_is_monotone_at_3_3():
    row = ladder(3)
    assert row["mask_blind"] >= row["mask_aware"] >= row["mask_blind_plus_memory"]
    assert row["mask_blind_plus_memory"] == 0.0          # memory fully de-aliases reserves
    assert row["mask_aware_no_common_legal_action"] == 0  # mask groups share a move-set
    # At (3,3) the mask already erases the (small) blind floor entirely.
    assert row["mask_blind"] > 0.0
    assert row["mask_aware"] == 0.0


def test_recorded_ladder_4_4():
    data = json.loads(RESULTS.read_text())
    assert data["provenance"]["config"]["feature"] == "hide_reserves"
    row = data["results"]["by_size"]["4_4"]
    assert row["states"] == 1357963
    assert abs(row["mask_blind"] - 0.0805) < 5e-4     # (board, cooldown) only
    assert abs(row["mask_aware"] - 0.0026) < 5e-4     # + legal-move mask (~30x cheaper)
    assert row["mask_blind_plus_memory"] == 0.0       # + destroyed-bead memory
    assert row["mask_aware_no_common_legal_action"] == 0
    # The load-bearing claim: the mask erases most of the cost, memory the rest.
    assert row["mask_aware"] < row["mask_blind"] / 10


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to recompute the (4,4) ladder (~90s)",
)
def test_recompute_ladder_4_4_matches_record():
    row = ladder(4)
    assert abs(row["mask_blind"] - 0.0805) < 5e-4
    assert abs(row["mask_aware"] - 0.0026) < 5e-4
    assert row["mask_blind_plus_memory"] == 0.0
