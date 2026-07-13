"""KimiAgent v1/v2 + best-response certification (Finding 8's crossover).

Locks, in order:
  * the frozen policies' mechanics on hand-built positions: peg priority,
    tier order, lethal-only removals, and the v1/v2 divergence position
    (v1's tier-local safety filter ignores a tower block that v2 takes);
  * Gate C at (3,3), re-solved live: v1 forced loss both seats, v2 certified
    DRAW both seats -- the strategy-complexity crossover's left edge;
  * (4,4) re-solves behind COLLAPSE3_SLOW=1: both policies forced losses;
  * recorded-JSON guards on best_response / kimi_gates / kimi_census so no
    bundle can ship contradicting the results files.
"""

import json
import os
from pathlib import Path

import pytest

from collapse3.agents import KimiAgentV1, KimiAgentV2
from collapse3.enumeration import reachable_states
from collapse3.game import (
    GameState,
    apply_move,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    orient,
)
from experiments.best_response import solve_best_response

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Policy mechanics on hand-built positions
# ---------------------------------------------------------------------------

def test_priority_centre_first_then_corner():
    v1, v2 = KimiAgentV1(), KimiAgentV2()
    s = empty_state(3, 3, 0)
    assert v1.choose(s) == ("place", 4)
    assert v2.choose(s) == ("place", 4)
    # Centre taken: first empty peg in priority order is corner 0.
    s2 = GameState(board=((), (), (), (), (0,), (), (), (), ()),
                   res=(2, 3), turn=1, cooldown=(0, 0))
    assert v1.choose(s2) == ("place", 0)
    assert v2.choose(s2) == ("place", 0)


def test_removals_are_lethal_only_when_placements_exist():
    """Scan all (3,3) decision states. v1: a removal chosen while a placement
    is legal must be an immediate win-in-1 (its tier walk stops at the first
    non-empty placement tier). v2 additionally reaches T3 removals through its
    cross-tier safety scan, so a non-lethal removal is allowed -- but only
    when it is SAFE and every placement before it was unsafe."""
    from collapse3.game import attrition_value

    def end_val(s):
        t = evaluate_terminal(s)
        if t is None and not get_legal_moves(s):
            t = attrition_value(s.board)
        return t

    v1, v2 = KimiAgentV1(), KimiAgentV2()
    v2_safe_removals = 0
    for state in reachable_states(empty_state(3, 3)):
        if evaluate_terminal(state) is not None:
            continue
        moves = get_legal_moves(state)
        if not moves or not any(m[0] == "place" for m in moves):
            continue

        chosen = v1.choose(state)
        if chosen[0] == "remove":
            t = end_val(apply_move(state, chosen))
            assert t is not None and orient(t, state.turn) > 0, (
                f"v1 non-lethal removal at {state}")

        chosen = v2.choose(state)
        if chosen[0] == "remove":
            t = end_val(apply_move(state, chosen))
            if t is not None and orient(t, state.turn) > 0:
                continue  # lethal, same as v1
            # Non-lethal: must itself be safe while all placements are unsafe.
            assert v2._safe(state, chosen), f"v2 unsafe non-lethal removal at {state}"
            assert not any(v2._safe(state, m) for m in moves if m[0] == "place"), (
                f"v2 removed while a safe placement existed at {state}")
            v2_safe_removals += 1
    assert v2_safe_removals > 0  # the escalation path is actually exercised


def test_v1_v2_divergence_tower_block():
    """The signature seam: P1 owns a two-bead tower on peg 0 (one bead from a
    vertical win). P0 has no win-in-1; empty pegs exist, so v1's first
    non-empty tier is T0 and its tier-local safety filter never reaches the
    T1 block -- it plays centre and loses to place-0. v2's cross-tier safety
    scan takes the block."""
    s = GameState(board=((1, 1), (), (), (), (), (), (), (), ()),
                  res=(3, 1), turn=0, cooldown=(0, 0))
    assert KimiAgentV1().choose(s) == ("place", 4)   # ignores the threat
    assert KimiAgentV2().choose(s) == ("place", 0)   # blocks the tower
    # And the ignored threat is real: after v1's move P1 wins with place 0.
    child = apply_move(s, ("place", 4))
    win = apply_move(child, ("place", 0))
    t = evaluate_terminal(win)
    assert t is not None and orient(t, 1) > 0


# ---------------------------------------------------------------------------
# Gate C at (3,3), re-solved live (fast: one-sided trees are tiny)
# ---------------------------------------------------------------------------

def test_gate_c_3_3_crossover_edge():
    # v1: forced loss from both seats.
    for seat in (0, 1):
        worst, _, _, _ = solve_best_response(KimiAgentV1(), 3, 3, seat)
        assert worst == -1, f"v1 seat P{seat} expected forced loss"
    # v2: certified exact drawing policy at (3,3), both seats.
    for seat in (0, 1):
        worst, _, _, _ = solve_best_response(KimiAgentV2(), 3, 3, seat)
        assert worst == 0, f"v2 seat P{seat} expected certified draw"


def test_gate_c_3_3_v1_exploit_is_short():
    worst, depth, _, _ = solve_best_response(KimiAgentV1(), 3, 3, 1)
    assert worst == -1 and depth == 5   # the bare-tower kill


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 to re-solve Gate C at (4,4)/(5,5)",
)
def test_gate_c_4_4_and_5_5_both_policies_lose():
    for r in (4, 5):
        for cls in (KimiAgentV1, KimiAgentV2):
            for seat in (0, 1):
                worst, _, _, _ = solve_best_response(cls(), r, r, seat)
                assert worst == -1, f"{cls.__name__} ({r},{r}) seat P{seat}"


# ---------------------------------------------------------------------------
# Recorded-JSON guards
# ---------------------------------------------------------------------------

def test_recorded_best_response_crossover_table():
    rows = json.loads((RESULTS / "best_response_latest.json").read_text())["results"]["rows"]
    table = {(r["policy"], tuple(r["reserves"]), r["policy_seat"]): r for r in rows}
    for seat in (0, 1):
        assert table[("kimi-v1", (3, 3), seat)]["worst_case"] == "forced loss"
        assert table[("kimi-v2", (3, 3), seat)]["worst_case"] == "draw"
        for r in (4, 5):
            assert table[("kimi-v1", (r, r), seat)]["worst_case"] == "forced loss"
            assert table[("kimi-v2", (r, r), seat)]["worst_case"] == "forced loss"
    # Shortest exploits: v1 5-6 plies everywhere; v2 (when it falls) 7-8.
    assert table[("kimi-v1", (4, 4), 0)]["shortest_exploit_plies"] == 6
    assert table[("kimi-v1", (4, 4), 1)]["shortest_exploit_plies"] == 5
    assert table[("kimi-v2", (4, 4), 0)]["shortest_exploit_plies"] == 8
    assert table[("kimi-v2", (4, 4), 1)]["shortest_exploit_plies"] == 7
    # Seat-P1 exploit at (3,3): value-preserving opponent play, v1 blunders.
    g = table[("kimi-v1", (3, 3), 1)]["exploit_grading"]
    opp_moves = [m for m in g["moves"] if m["mover"] == "opponent"]
    assert all(m["true_game"] == "optimal" for m in opp_moves)
    assert g["first_policy_positive_regret_ply"] is not None
    # Seat-P0 exploit at (4,4): routes through a true-game opponent BLUNDER
    # (Finding 3's mechanism).
    g0 = table[("kimi-v1", (4, 4), 0)]["exploit_grading"]
    assert any(m["mover"] == "opponent" and m["true_game"].startswith("blunder")
               for m in g0["moves"])


def test_recorded_gate_b_juxtaposition():
    """Gate B's clean-looking record must never ship without Gate C: guard the
    numbers that make the juxtaposition. At eps=0.05 v1 loses 1/1200 with
    crit-opt 0.993 -- while carrying a 5-ply forced refutation."""
    res = json.loads((RESULTS / "kimi_gates_latest.json").read_text())["results"]
    assert res["gate_a"]["pass"] is True
    b = res["gate_b"]
    assert b["kimi-v1@eps=0.05"]["games"] == 1200
    assert b["kimi-v1@eps=0.05"]["losses"] == 1
    assert b["kimi-v1@eps=0.25"]["losses"] == 9
    assert b["kimi-v1@eps=0.05"]["crit_opt_rate"] >= 0.99
    assert b["kimi-v2@eps=0.25"]["losses"] == 6


def test_recorded_census_v2_beats_v1_where_it_matters():
    res = json.loads((RESULTS / "kimi_census_latest.json").read_text())["results"]
    v1_44, v2_44 = res["kimi-v1@(4,4)"], res["kimi-v2@(4,4)"]
    assert v1_44["decision_states"] == 477960
    # v2 halves v1's critical regret at (4,4) yet still carries a forced loss.
    assert v2_44["mean_wdl_regret_critical"] < v1_44["mean_wdl_regret_critical"]
    assert v1_44["optimal_rate_critical"] == pytest.approx(0.9206, abs=1e-4)
    assert v2_44["optimal_rate_critical"] == pytest.approx(0.9570, abs=1e-4)
    # v2's residual (4,4) errors concentrate in the land-grab phase (beads
    # 3-4), where forks are prepared just past its one-reply safety horizon.
    phases = res["kimi-v2@(4,4)"]["by_phase"]
    worst_phase = max(phases, key=lambda k: phases[k]["mean_wdl_regret"])
    assert worst_phase == "land-grab(3-4)"
