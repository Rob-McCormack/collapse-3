#!/usr/bin/env python3
"""Stochastic-removal probes for Collapse3 (rule forks, exact expectiminimax).

Modes per seat:
  'none' : deliberate removal (peg AND bead chosen)  -- the shipped game
  'bead' : peg chosen deliberately, bead uniform-random among legal targets
  'full' : no intelligence: 'declare collapse', one uniform-random (peg, bead)

The game tree is a DAG: (reserves_total, beads_on_board) strictly decreases
lexicographically with every action, so memoized recursion is exact.

Payoffs (terminal score transforms, P0 perspective):
  'ev'        : raw engine score (+100/+10/0/-10/-100) -> expected value
  'p_win'     : 1 if P0 wins else 0    -> P0's secured win probability
  'p_notlose' : 1 if P0 doesn't lose   -> 1 - P1's secured win probability
  'wdl'       : +1 win / 0 draw / -1 loss

SCOPE: a stochastic fork of the three-peg sibling (plus full-board small
sizes). Expected values, NOT certificates. Never quote as facts about Collapse3.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collapse3 import solver as sv
from collapse3.game import (empty_state, get_legal_moves, apply_move,
                            evaluate_terminal, attrition_value, placement_pegs)

PAYOFFS = {
    'ev': lambda s: float(s),
    'p_win': lambda s: 1.0 if s > 0 else 0.0,
    'p_notlose': lambda s: 1.0 if s >= 0 else 0.0,
    'wdl': lambda s: 1.0 if s > 0 else (0.0 if s == 0 else -1.0),
}

def make_solver(mode0, mode1, payoff_name):
    payoff = PAYOFFS[payoff_name]
    memo = {}

    def V(state):
        hit = memo.get(state)
        if hit is not None:
            return hit
        t = evaluate_terminal(state)
        if t is not None:
            memo[state] = payoff(t); return memo[state]
        moves = get_legal_moves(state)
        if not moves:
            memo[state] = payoff(attrition_value(state.board)); return memo[state]
        p = state.turn
        mode = mode0 if p == 0 else mode1
        action_values = []
        by_peg, all_targets = {}, []
        for m in moves:
            if m[0] == 'place':
                action_values.append(V(apply_move(state, m)))
            else:
                by_peg.setdefault(m[1], []).append(m[2])
                all_targets.append((m[1], m[2]))
        if mode == 'none':
            for peg, zs in by_peg.items():
                for z in zs:
                    action_values.append(V(apply_move(state, ('remove', peg, z))))
        elif mode == 'bead':
            for peg, zs in by_peg.items():
                ev = sum(V(apply_move(state, ('remove', peg, z))) for z in zs) / len(zs)
                action_values.append(ev)
        elif all_targets:  # 'full'
            ev = sum(V(apply_move(state, ('remove', pg, z)))
                     for pg, z in all_targets) / len(all_targets)
            action_values.append(ev)
        memo[state] = max(action_values) if p == 0 else min(action_values)
        return memo[state]

    V.memo = memo
    return V

class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False

def solve_root(r, mode0, mode1, payoff_name='ev', pegs=None):
    V = make_solver(mode0, mode1, payoff_name)
    ctx = placement_pegs(pegs) if pegs is not None else _nullctx()
    with ctx:
        v = V(empty_state(r, r))
    return v, len(V.memo)

THREE_PEG = (0, 1, 2)

print("=== 0. Trust anchors (must match the published repo exactly) ===")
sv.reset_search_state()
v1414 = sv.game_value(empty_state(14, 14))
print(f"(14,14) root: {v1414:+d}  nodes {sv.nodes_visited:,}  (expect +100, ~96K nodes)")
for r in range(1, 8):
    sv.reset_search_state()
    print(f"({r},{r}) root {sv.game_value(empty_state(r, r)):+d}", end="  ")
print(" (expect +0 r=1..5, +100 r>=6)")
for r in range(2, 8):
    sv.reset_search_state()
    with placement_pegs(THREE_PEG):
        v3 = sv.game_value(empty_state(r, r))
    print(f"three-peg ({r},{r}) {v3:+d}", end="  ")
print(" (expect +0 at (2,2), +10 from (3,3))")

print("\n=== A. Both-sided bead-random, three-peg, full decomposition ===")
print("size      det      bead-EV   p_win  p_notlose  wdl")
for r in range(3, 11):
    v_det, _ = solve_root(r, 'none', 'none', 'ev', pegs=THREE_PEG)
    v_b, n   = solve_root(r, 'bead', 'bead', 'ev', pegs=THREE_PEG)
    pw, _    = solve_root(r, 'bead', 'bead', 'p_win', pegs=THREE_PEG)
    pnl, _   = solve_root(r, 'bead', 'bead', 'p_notlose', pegs=THREE_PEG)
    w, _     = solve_root(r, 'bead', 'bead', 'wdl', pegs=THREE_PEG)
    print(f"({r},{r})  {v_det:+.1f}  {v_b!r:>8}  {pw}  {pnl}  {w!r}  [{n} states]")

print("\n=== B. Both-sided full-random ('declare collapse'), three-peg ===")
for r in range(3, 9):
    v_f, _ = solve_root(r, 'full', 'full', 'ev', pegs=THREE_PEG)
    print(f"({r},{r})  full-random EV = {v_f!r}")

print("\n=== C. One-sided probes (whose precision?), three-peg ===")
for r in range(4, 9):
    v_none, _ = solve_root(r, 'none', 'none', 'ev', pegs=THREE_PEG)
    v_p0r, _  = solve_root(r, 'bead', 'none', 'ev', pegs=THREE_PEG)
    v_p1r, _  = solve_root(r, 'none', 'bead', 'ev', pegs=THREE_PEG)
    print(f"({r},{r})  both precise {v_none:+.1f} | P0 imprecise {v_p0r!r} | P1 imprecise {v_p1r!r}")
pw, _  = solve_root(7, 'bead', 'none', 'p_win', pegs=THREE_PEG)
pnl, _ = solve_root(7, 'bead', 'none', 'p_notlose', pegs=THREE_PEG)
print(f"(7,7) P0 imprecise decomposed: P0 secures win {pw}, >=draw {pnl} -> "
      f"coin flip: 0.5*(+10 attrition) + 0.5*(-100 line loss) = -45")

print("\n=== D. Full board, both-sided bead (small sizes) ===")
for r in (2, 3, 4):
    v_det, _ = solve_root(r, 'none', 'none')
    v_b, n   = solve_root(r, 'bead', 'bead')
    print(f"({r},{r})  det {v_det:+.1f}  bead-EV {v_b!r}  [{n} states] "
          f"(expect states 4,051 / 97,093 / 1,357,963)")
