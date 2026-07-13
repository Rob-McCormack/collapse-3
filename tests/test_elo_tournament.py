"""Guards on the Elo-mirage tournament (FAQ #10 / Finding 1 cross-ref).

The claims locked here, against the shipped results JSON:
  * Elo ranks a certifiably exploitable rulebook (kimi-v2, forced loss at
    (4,4)) ABOVE the perfect player;
  * the perfect player is the only agent with zero losses and zero regret;
  * the regret column orders the roster differently from (and more honestly
    than) the Elo column at the top;
  * the myopic-vs-kimi-v1 seat split (Elo's worst single prediction).
Plus a fast sanity check of the rating fit itself.
"""

import json
from pathlib import Path

from experiments.elo_tournament import expected, fit_elo

RESULTS = Path(__file__).resolve().parent.parent / "results" / "elo_tournament_latest.json"


def _table():
    data = json.loads(RESULTS.read_text())["results"]
    return {row["agent"]: row for row in data["table"]}, data["head_to_head"]


def test_fit_elo_recovers_a_known_ordering():
    # A beats B 90/100, B beats C 90/100, A beats C 45/50: ratings must be
    # strictly ordered, and the fitted A-B expected score must sit near the
    # empirical 0.9 (shrunk slightly toward 1/2 by the anchor prior and the
    # joint fit with the A-C record).
    scores = {"A": 90 + 45, "B": 10 + 90, "C": 10 + 5}
    games = {("A", "B"): 100, ("B", "A"): 100,
             ("B", "C"): 100, ("C", "B"): 100,
             ("A", "C"): 50, ("C", "A"): 50}
    r = fit_elo(scores, games)
    assert r["A"] > r["B"] > r["C"]
    assert 0.80 < expected(r["A"], r["B"]) < 0.92


def test_recorded_elo_ranks_exploitable_rulebook_above_perfect_player():
    table, _ = _table()
    assert table["kimi-v2"]["elo"] > table["optimal"]["elo"]
    assert table["kimi-v2"]["worst_case"].startswith("forced loss")
    assert table["optimal"]["worst_case"] == ">= draw (theorem)"


def test_recorded_only_optimal_is_lossless_and_regret_free():
    table, _ = _table()
    assert table["optimal"]["losses"] == 0
    assert table["optimal"]["mean_wdl_regret"] == 0.0
    for agent, row in table.items():
        if agent != "optimal":
            assert row["losses"] > 0
            assert row["mean_wdl_regret"] > 0.0


def test_recorded_regret_orders_the_top_differently_from_elo():
    table, _ = _table()
    by_elo = sorted(table, key=lambda a: -table[a]["elo"])
    by_regret = sorted(table, key=lambda a: table[a]["mean_wdl_regret"])
    assert by_elo[0] == "kimi-v2" and by_regret[0] == "optimal"
    # ...while the bottom of the table agrees (both metrics see random as worst).
    assert by_elo[-1] == by_regret[-1] == "random"


def test_recorded_myopic_seat_split_vs_kimi_v1():
    """Elo's worst prediction: a 286-point gap implies kimi-v1 scores ~84%
    against myopic; the recorded head-to-head is a pure seat split (each side
    wins all its first-player games), i.e. 50%."""
    table, h2h = _table()
    gap = table["kimi-v1"]["elo"] - table["myopic"]["elo"]
    assert gap > 250
    rec = h2h["kimi-v1"]["myopic"]
    assert rec["w"] == 50 and rec["l"] == 50 and rec["d"] == 0
