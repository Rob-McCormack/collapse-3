"""Exact value-of-information: *when* should an agent acquire a missing feature?

Finding 4 / ``masked_floor`` ask a static question: what does a fixed missing
feature *cost*? This experiment asks the adaptive one: given a coarse interface,
in which states should an optimal agent *choose to query* a richer observation
before acting -- and how little querying buys back the whole irreducible floor?

Because the game is solved, the answer is exact. For each coarse observation
group we compare two options:

  * act on the coarse observation (pay that group's aliasing regret), or
  * query the finer observation (split the group into its finer sub-groups,
    each acting on its own best common move) and pay their regret plus a query
    cost.

The per-group minimum is an exact *information policy*: an oracle labelling of
where the coarse representation is genuinely insufficient. It is the same exact,
uniform-over-decision, opponent-independent machinery as the aliasing floor
(``collapse3.aliasing``), lifted from "one feature, always missing" to "one
feature, acquired on demand."

Two axes, each isolating one information good (refine strictly refines the base
partition, so "query" only ever splits groups):

  mask   reserves already hidden; base = (board,turn,cooldown); query the
         legal-action mask.  (Illustrative: the mask is normally free.)
  reserve  the mask is already in hand (what the trained QAgent sees); base =
         (board,turn,cooldown,mask); query the reserve counts (-> full state).
         This is the load-bearing axis: on the mask-aware interface reserves
         still carry a nonzero floor from (4,4) up, and we measure how sparse
         the states that need them are.

This is an exact, game-based instance of *value of information* (Howard 1966)
and metareasoning about when to sense/compute (Russell & Wefald 1991); the
contribution here is exactness on a fully solved game, not the concept.

Run:  python -m experiments.info_policy 3 4        # default; ~1-2 min at (4,4)
      python -m experiments.info_policy 3 4 5      # add (5,5); heavy (~12.7M states)
"""

import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from collapse3.enumeration import solve_all, wdl
from collapse3.game import (
    GameState,
    apply_move,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
)
from experiments._provenance import announce, write_result

NAME = "info_policy"

# Cost per query, in WDL-regret units per queried decision (regret is in [0, 2]).
COSTS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 1.0)

Decision = Tuple[GameState, Dict, int]  # (state, action->wdl value, best value)
ObsFn = Callable[[GameState], Tuple]


def _mask(s: GameState) -> Tuple:
    return tuple(sorted(get_legal_moves(s)))


# Base / refine observation pairs per axis. ``refine`` must strictly refine
# ``base`` (add a field), so querying can only split a base group, never merge.
AXES: Dict[str, Tuple[ObsFn, ObsFn]] = {
    "mask": (
        lambda s: (s.board, s.turn, s.cooldown),
        lambda s: (s.board, s.turn, s.cooldown, _mask(s)),
    ),
    "reserve": (
        lambda s: (s.board, s.turn, s.cooldown, _mask(s)),
        lambda s: (s.board, s.res, s.turn, s.cooldown, _mask(s)),
    ),
}


def decisions_of(memo: Dict[GameState, int]) -> List[Decision]:
    out: List[Decision] = []
    for s in memo:
        if evaluate_terminal(s) is not None:
            continue
        moves = get_legal_moves(s)
        if not moves:
            continue
        mover = s.turn
        avals = {m: wdl(memo[apply_move(s, m)], mover) for m in moves}
        out.append((s, avals, max(avals.values())))
    return out


def group_regret(members: List[Decision]) -> float:
    """Total WDL regret of the best action legal in every member (charity: 0)."""
    candidates = set()
    for _, avals, _ in members:
        candidates |= set(avals)
    best = None
    for cand in candidates:
        tot, ok = 0.0, True
        for _, avals, vstar in members:
            if cand not in avals:
                ok = False
                break
            tot += vstar - avals[cand]
        if ok and (best is None or tot < best):
            best = tot
    return best if best is not None else 0.0


def info_policy(decisions: List[Decision], base_fn: ObsFn, refine_fn: ObsFn) -> dict:
    n = len(decisions)
    base_groups: Dict[Tuple, List[Decision]] = defaultdict(list)
    for d in decisions:
        base_groups[base_fn(d[0])].append(d)

    # Per base group: regret if we act coarse, vs regret if we query (split).
    per_group = []  # (size, no_query_regret, query_regret)
    base_total = refined_total = 0.0
    beneficial_groups = beneficial_states = 0
    for members in base_groups.values():
        no_query = group_regret(members)
        sub: Dict[Tuple, List[Decision]] = defaultdict(list)
        for d in members:
            sub[refine_fn(d[0])].append(d)
        query = sum(group_regret(m) for m in sub.values())
        per_group.append((len(members), no_query, query))
        base_total += no_query
        refined_total += query
        if query < no_query - 1e-12:
            beneficial_groups += 1
            beneficial_states += len(members)

    rows = []
    for c in COSTS:
        residual = queried_states = 0.0
        for size, no_query, query in per_group:
            # Query iff it strictly lowers total objective for this group.
            if query + c * size < no_query - 1e-12:
                residual += query
                queried_states += size
            else:
                residual += no_query
        residual /= n
        qfrac = queried_states / n
        rows.append({
            "cost": c,
            "query_state_fraction": qfrac,
            "residual_regret": residual,
            "total_objective": residual + c * qfrac,
        })

    return {
        "decision_states": n,
        "base_groups": len(base_groups),
        "base_floor": base_total / n,
        "refined_floor": refined_total / n,
        "beneficial_query_groups": beneficial_groups,
        "beneficial_query_states": beneficial_states,
        "beneficial_query_state_fraction": beneficial_states / n,
        "rows": rows,
    }


def main(sizes=(3, 4)) -> None:
    by_size: Dict[str, dict] = {}
    for r in sizes:
        t0 = time.time()
        memo = solve_all(empty_state(r, r))
        decisions = decisions_of(memo)
        axes = {name: info_policy(decisions, base, refine)
                for name, (base, refine) in AXES.items()}
        by_size[f"{r}_{r}"] = {"states": len(memo), "axes": axes}
        print(f"\n({r},{r})  states={len(memo):,}  decisions={len(decisions):,}"
              f"  [{time.time()-t0:.0f}s]")
        for name, res in axes.items():
            print(f"  {name:>8}: base_floor={res['base_floor']:.7f} -> "
                  f"refined={res['refined_floor']:.7f}   "
                  f"query {res['beneficial_query_states']}/{res['decision_states']} "
                  f"states = {res['beneficial_query_state_fraction']*100:.3f}%")

    payload = {"sizes": list(sizes), "costs": list(COSTS), "by_size": by_size}
    path = write_result(NAME, {"sizes": list(sizes)}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else (3, 4))
