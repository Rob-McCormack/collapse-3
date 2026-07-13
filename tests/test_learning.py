"""Learning-module tests: observations, opponent, training determinism."""

import json
import random
from pathlib import Path

from collapse3.enumeration import solve_all
from collapse3.game import empty_state, get_legal_moves
from collapse3.learning import (
    OBSERVATIONS,
    QAgent,
    advance_to_agent,
    optimal_opponent,
    train_q,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_observation_lossiness():
    s = empty_state(2, 2)
    assert OBSERVATIONS["full"](s) != OBSERVATIONS["hide_cooldown"](s)
    assert len(OBSERVATIONS["hide_both"](s)) < len(OBSERVATIONS["full"](s))


def test_optimal_opponent_returns_legal_moves():
    root = empty_state(2, 2)
    memo = solve_all(root)
    opp = optimal_opponent(memo, epsilon=0.0, rng=random.Random(0))
    assert opp(root) in get_legal_moves(root)


def test_training_is_deterministic_and_agent_plays_legally():
    root = empty_state(2, 2)
    memo = solve_all(root)
    opp1 = optimal_opponent(memo, 0.25, random.Random(1))
    opp2 = optimal_opponent(memo, 0.25, random.Random(1))
    Qa = train_q(root, "full", 500, opp1, seed=7)
    Qb = train_q(root, "full", 500, opp2, seed=7)
    assert Qa == Qb  # same seeds -> identical tables

    agent = QAgent(Qa, "full")
    assert agent.choose(root) in get_legal_moves(root)


def test_advance_to_agent_reaches_agent_or_terminal():
    root = empty_state(2, 2)
    memo = solve_all(root)
    opp = optimal_opponent(memo, 0.0, random.Random(0))
    state, reward = advance_to_agent(root, 0, opp)
    # P0 moves first at the root, so it's immediately the agent's turn.
    assert reward is None
    assert state.turn == 0


def test_q_table_covers_only_its_training_seat():
    """The seat bug behind the withdrawn Finding-2 claim (external review).

    A Q-table trained at seat 0 contains no seat-1 decision keys (the obs key
    includes the turn), so playing it as P1 exercises the untrained
    first-legal-move fallback. Head-to-head experiments must therefore use one
    table per seat -- this locks the fact that makes that mandatory.
    """
    root = empty_state(2, 2)
    memo = solve_all(root)
    Q0 = train_q(root, "full", 500, optimal_opponent(memo, 0.25, random.Random(1)),
                 agent_seat=0, seed=7)
    Q1 = train_q(root, "full", 500, optimal_opponent(memo, 0.25, random.Random(1)),
                 agent_seat=1, seed=7)
    # obs key = (board, res, turn, cooldown); Q key = (obs_key, move)
    assert {obs_key[2] for (obs_key, _m) in Q0} == {0}
    assert {obs_key[2] for (obs_key, _m) in Q1} == {1}


def test_recorded_head_to_head_is_seat_matched():
    """Guard the corrected Finding-2 numbers against the shipped results file."""
    data = json.loads((RESULTS / "oneply_vs_learned_latest.json").read_text())
    res = data["results"]
    assert "q_entries_seat1" in res      # per-seat training actually happened
    h2h = res["head_to_head"]["oneply_P0_vs_learned_P1"]
    # Corrected result: the seat-1 learner loses some games but not all of them.
    assert h2h["P0_wins"] == 58 and h2h["draws"] == 123 and h2h["P0_losses"] == 19
    assert res["audit"]["learned_tabular_q_seat1"]["mean_regret"] < 0.05
