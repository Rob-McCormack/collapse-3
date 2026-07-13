"""Cross-validate the shipped rules engine against the clean-room reference.

Three tiers:
  * spot checks on hand-built positions (fast, default suite);
  * full agreement on every reachable state at (3,3) — legal moves, every
    transition, terminal score, and stuck-state attrition end;
  * random deep playouts at (4,4) and (5,5) (slow tier).

Any mismatch aborts with a concrete state + move diff — the whole point is to
catch rules drift before it poisons the solver's theorems.
"""

import os
import random

import pytest

from collapse3.enumeration import reachable_states
from collapse3.game import (
    GameState,
    apply_move,
    attrition_value,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
)
from collapse3.reference_engine import (
    RefState,
    ref_apply,
    ref_empty,
    ref_end_if_stuck,
    ref_has_line,
    ref_legal_moves,
    ref_terminal,
)


def _to_ref(state: GameState) -> RefState:
    return RefState(state.board, state.res, state.turn, state.cooldown)


def _from_ref(ref: RefState) -> GameState:
    return GameState(ref.board, ref.reserves, ref.side, ref.removed_last_turn)


def _assert_state(state: GameState) -> None:
    """Compare one state: terminal score, legal moves, stuck attrition, transitions."""
    ref = _to_ref(state)
    t_main = evaluate_terminal(state)
    t_ref = ref_terminal(ref)
    assert t_main == t_ref, f"terminal mismatch at {state}: main={t_main} ref={t_ref}"

    lm = set(get_legal_moves(state))
    lr = set(ref_legal_moves(ref))
    assert lm == lr, f"legal-move mismatch at {state}: only-main={lm - lr} only-ref={lr - lm}"

    if not lm:
        if t_main is None:
            stuck_main = attrition_value(state.board)
            stuck_ref = ref_end_if_stuck(ref)
            assert stuck_main == stuck_ref, (
                f"stuck-state attrition mismatch at {state}: main={stuck_main} ref={stuck_ref}"
            )
        return

    for m in lm:
        child_main = apply_move(state, m)
        child_ref = ref_apply(ref, m)
        assert child_main == _from_ref(child_ref), (
            f"transition mismatch on {m} from {state}: main={child_main} ref={child_ref}"
        )


def test_spot_checks_match_reference():
    """Hand-built positions from test_game.py, checked both ways."""
    E = tuple()

    def board(**kw):
        b = [E] * 9
        for k, v in kw.items():
            b[int(k[1:])] = tuple(v)
        return tuple(b)

    assert ref_has_line(board(p0=[0, 0, 0]), 0)
    assert not ref_has_line(board(p0=[0, 0, 0]), 1)
    assert ref_has_line(board(p0=[0], p1=[1, 0], p2=[1, 1, 0]), 0)

    oops = GameState(board(p0=[1], p1=[1], p2=[1]), (1, 1), 1, (False, True))
    assert evaluate_terminal(oops) == ref_terminal(_to_ref(oops)) == -100

    b = board(p0=[0, 0, 0], p3=[1], p4=[1], p5=[1])
    sim_p0 = GameState(b, (1, 1), 1, (False, False))
    sim_p1 = GameState(b, (1, 1), 0, (False, False))
    assert evaluate_terminal(sim_p0) == ref_terminal(_to_ref(sim_p0)) == 100
    assert evaluate_terminal(sim_p1) == ref_terminal(_to_ref(sim_p1)) == -100

    st = GameState(board(p0=[0, 1], p1=[0]), (5, 5), 0, (False, False))
    _assert_state(st)


def test_full_enumeration_agreement_at_3_3():
    root = empty_state(3, 3)
    states = reachable_states(root)
    assert len(states) > 90_000
    for i, state in enumerate(states):
        _assert_state(state)
        if i and i % 25_000 == 0:
            print(f"  checked {i:,} / {len(states):,} states", flush=True)


def _random_playout(r: int, seed: int, max_plies: int = 400) -> None:
    rng = random.Random(seed)
    state = empty_state(r, r)
    for _ in range(max_plies):
        _assert_state(state)
        t = evaluate_terminal(state)
        if t is not None:
            assert t == ref_terminal(_to_ref(state))
            return
        moves = get_legal_moves(state)
        if not moves:
            assert attrition_value(state.board) == ref_end_if_stuck(_to_ref(state))
            return
        state = apply_move(state, rng.choice(moves))
    pytest.fail(f"playout exceeded {max_plies} plies")


@pytest.mark.parametrize("seed", range(20))
def test_random_playouts_at_4_4(seed: int):
    _random_playout(4, seed)


@pytest.mark.skipif(
    os.environ.get("COLLAPSE3_SLOW") != "1",
    reason="set COLLAPSE3_SLOW=1 for (5,5) reference playouts",
)
@pytest.mark.parametrize("seed", range(10))
def test_random_playouts_at_5_5(seed: int):
    _random_playout(5, seed, max_plies=500)
