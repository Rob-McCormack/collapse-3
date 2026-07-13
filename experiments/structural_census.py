"""Exhaustive structural census of Collapse3 at reserves (r, r).

An exact, whole-state-space audit that answers: where does the game's
difficulty and richness actually live? Reported:

  1. Value landscape       -- split of states across line-win / attrition / draw.
  2. Opening theory        -- exact value of each first move (symmetry sanity).
  3. Is the Collapse load-bearing -- removal strictly best / only-optimal states.
  4. Oops census           -- states with an instant-loss removal.
  5. Opponent-cooldown aliasing -- does the opponent's cooldown ever change your
                              optimal move set?
  6. Tempo pressure        -- states where the mover would benefit from a
                              *hypothetical* pass. NOTE: passing is NOT a legal
                              action under the shipped immediate-end rules; this
                              is a diagnostic of tempo/zugzwang only.

Run:  python -m experiments.structural_census        # default r=4 (~1-2 min)
      python -m experiments.structural_census 3       # smaller, fast
"""

import random
import sys

from collapse3.enumeration import reachable_states, solve_all, wdl
from collapse3.game import (
    GameState,
    apply_move,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    orient,
)
from collapse3.solver import game_value
from experiments._provenance import announce, write_result

NAME = "structural_census"


def hypothetical_pass(s: GameState) -> GameState:
    """The state if the mover could pass (illegal; diagnostic only): turn flips
    and the mover's cooldown clears."""
    cd = list(s.cooldown)
    cd[s.turn] = False
    return GameState(s.board, s.res, 1 - s.turn, tuple(cd))


def is_oops_removal(state: GameState, move) -> bool:
    """True if this removal immediately ends the game as a loss for the mover
    (an Oops: the gravity cascade completes only the opponent's line)."""
    child = apply_move(state, move)
    t = evaluate_terminal(child)
    return t is not None and orient(t, state.turn) < 0


def main(r: int = 4) -> None:
    root = empty_state(r, r)
    reachable = reachable_states(root)
    memo = solve_all(root)

    total = len(reachable)
    labels = {100: "P0 line-win", 10: "P0 attrition-win", 0: "draw",
              -10: "P1 attrition-win", -100: "P1 line-win"}

    # 1. value landscape
    dist = {}
    for s in reachable:
        dist[memo[s]] = dist.get(memo[s], 0) + 1
    landscape = {labels[v]: dist.get(v, 0) for v in (100, 10, 0, -10, -100)}

    # 2. opening theory
    openings = {peg: memo[apply_move(root, ('place', peg))] for peg in range(9)}

    # 3/4/5 single pass over decision states
    n_dec = n_with_removal = removal_strictly_best = 0
    removal_only_optimal = removal_among_optimal = 0
    oops_available = all_removals_oops = 0
    opp_cd_groups = {}
    for s in reachable:
        if evaluate_terminal(s) is not None:
            continue
        moves = get_legal_moves(s)
        if not moves:
            continue
        n_dec += 1
        p = s.turn
        placements = [m for m in moves if m[0] == 'place']
        removals = [m for m in moves if m[0] == 'remove']
        vals = {m: orient(memo[apply_move(s, m)], p) for m in moves}
        vstar = max(vals.values())
        opt = {m for m, u in vals.items() if u == vstar}

        if removals:
            n_with_removal += 1
            best_rem = max(vals[m] for m in removals)
            best_pla = max((vals[m] for m in placements), default=-10 ** 9)
            if best_rem > best_pla:
                removal_strictly_best += 1
            if any(m in opt for m in removals):
                removal_among_optimal += 1
                if all(m[0] == 'remove' for m in opt):
                    removal_only_optimal += 1
            oops = [m for m in removals if is_oops_removal(s, m)]
            if oops:
                oops_available += 1
                if len(oops) == len(removals):
                    all_removals_oops += 1

        key = (s.board, s.res, s.turn, s.cooldown[p])  # opponent cooldown hidden
        opp_cd_groups.setdefault(key, set()).add(frozenset(opt))

    opp_cd_aliased = sum(1 for g in opp_cd_groups.values() if len(g) > 1)

    # 6. tempo pressure (hypothetical pass) over a sample
    decision_states = [s for s in reachable
                       if evaluate_terminal(s) is None and get_legal_moves(s)]
    rng = random.Random(11)
    sample = rng.sample(decision_states, min(20000, len(decision_states)))
    tempo = 0
    for s in sample:
        p = s.turn
        # The hypothetical-pass state is off-tree, so value it with the solver.
        if orient(game_value(hypothetical_pass(s)), p) > orient(memo[s], p):
            tempo += 1

    payload = {
        "reserves": [r, r],
        "reachable_states": total,
        "decision_states": n_dec,
        "value_landscape": landscape,
        "value_landscape_pct": {k: round(100 * v / total, 2) for k, v in landscape.items()},
        "openings": openings,
        "openings_all_equal": len(set(openings.values())) == 1,
        "collapse": {
            "with_removal_available": n_with_removal,
            "removal_strictly_best": removal_strictly_best,
            "removal_among_optimal": removal_among_optimal,
            "removal_only_optimal": removal_only_optimal,
        },
        "oops": {
            "states_with_oops_removal": oops_available,
            "states_all_removals_oops": all_removals_oops,
        },
        "opponent_cooldown_aliasing": {
            "groups": len(opp_cd_groups),
            "aliased_groups": opp_cd_aliased,
        },
        "tempo_pressure_sample": {
            "sampled": len(sample),
            "would_benefit_from_pass": tempo,
            "pct": round(100 * tempo / len(sample), 3) if sample else 0.0,
            "note": "pass is a hypothetical diagnostic; not a legal action",
        },
    }

    print(f"Structural census, reserves ({r},{r}) -- {total:,} reachable states")
    print("=" * 68)
    print("1. VALUE LANDSCAPE")
    for k in ("P0 line-win", "P0 attrition-win", "draw", "P1 attrition-win", "P1 line-win"):
        print(f"   {k:<18}: {landscape[k]:>10,}  ({payload['value_landscape_pct'][k]}%)")
    print(f"\n2. OPENINGS all equal by symmetry: {payload['openings_all_equal']}  "
          f"(value {next(iter(openings.values()))})")
    c = payload["collapse"]
    print(f"\n3. COLLAPSE load-bearing: removal available in {c['with_removal_available']:,} states; "
          f"strictly best in {c['removal_strictly_best']:,}; only-optimal in {c['removal_only_optimal']:,}")
    print(f"\n4. OOPS: {payload['oops']['states_with_oops_removal']:,} states with an instant-loss removal")
    print(f"\n5. OPP-COOLDOWN aliasing: {opp_cd_aliased:,} groups where hiding it "
          f"could matter (of {len(opp_cd_groups):,})")
    print(f"\n6. TEMPO (hypothetical pass): {payload['tempo_pressure_sample']['pct']}% of "
          f"{len(sample):,} sampled states would benefit from passing")

    path = write_result(NAME, {"reserves": [r, r]}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
