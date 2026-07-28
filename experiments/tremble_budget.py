"""How many opponent mistakes must an evaluation permit before it certifies?

Finding 17 (`experiments/evaluation_equivalence.py`) contrasts two extremes: an
evaluator whose opponent plays only value-optimal moves inspects as few as 10
candidate decisions and certifies nothing, while an evaluator whose opponent may
play ANY legal move rules the forced loss out. The all-legal end of that contrast
is not a protocol anyone can run. This experiment fills in the space between.

    TREMBLE BUDGET m: the evaluation opponent plays zero-regret moves but may
    make at most m value-dropping moves ("blunders") per game.

`support(m)` is the set of candidate decisions such an opponent can expose;
certification asks whether pinning the candidate to the reference policy on that
support still leaves a forced loss compatible (`compatible_extreme` semantics,
unchanged from Finding 17). m=0 is Gate C's optimal-only opponent; m=infinity is
bounded above by Gate C's all-legal opponent.

WHAT IS CLAIMED
    (i)  Under WDL grading, ONE blunder is enough: support(1) equals the
         all-legal support as a SET, and every cell that Gate C left as a
         certified forced loss flips to certified at m=1.
    (ii) Grading the opponent on the raw 5-class ladder (100/10/0/-10/-100)
         instead of WDL puts eight states beyond the reach of a single blunder,
         at BOTH sizes solved here. At (3,3) that costs nothing -- certification
         still flips at m=1. At (4,4) seat 0 it becomes a cliff: m=1 still
         leaves a certified forced loss and certification waits for m=2. The
         CLIFF is the artifact of scale mismatch between the grader and the
         candidate's objective, and a class-aware reference policy (V2) removes
         it -- along with the eight, which leave the support universe entirely.
         V2 does NOT remove the eight at (3,3). Scale-matching dissolves the
         gating, not the second blunder as such.

WHAT IS NOT CLAIMED
    That the budget is bounded by the number of value classes below the root in
    general. Every cell above starts from the opening, which is a draw at every
    size solved here, so a class-count rule and the observed budget agree there
    whether or not one causes the other. `conjecture_scan` re-roots the same
    machinery at non-drawn states to separate them -- exhaustively at (3,3) --
    and the unconditional rule FAILS at (4,4) under a mismatched
    grader/reference pair. What survives is scoped to matched pairs; see
    `CONJECTURE`, which records the refutations rather than the reworded rule.

*** ALL ABSOLUTE COUNTS ARE REFERENCE-POLICY-DEPENDENT ***
A support set is the set of decisions ONE frozen candidate is driven through, so
every count here moves if the reference policy moves -- demonstrated directly by
the reversed-tie-break row, which keeps the flip-at-m=1 pattern while the counts
change. Only the PATTERNS are claimed: set equalities, the budget at which
certification flips, and the maximum over min-blunder distances. Read no
significance into a particular integer.

Certification itself is always WDL (a forced loss is a forced loss); the grading
choice governs only which opponent moves count as blunders.
"""

import sys
import time
from collections import Counter, deque
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import experiments.evaluation_equivalence as ee
from collapse3.enumeration import solve_all, wdl
from collapse3.game import (ATTRITION_LOSS, GameState, Move, apply_move,
                            empty_state, get_legal_moves, orient, tt_key)
from experiments._provenance import announce, write_result

NAME = "tremble_budget"

sys.setrecursionlimit(1_000_000)

# Reference policies. V1 is Finding 17's frozen policy, reused unchanged; the
# other two exist only to show what the counts do when the reference moves.
POLICY_V1 = "CANONICAL_POLICY_V1"           # WDL-optimal, canonical_move_key
POLICY_V2 = "CANONICAL_POLICY_V2"           # class-optimal, canonical_move_key
POLICY_V1R = "CANONICAL_POLICY_V1_REVERSED"  # WDL-optimal, reversed tie-break

GRADE_WDL = "wdl"
GRADE_CLASS = "class5"

BUDGETS: Tuple[int, ...] = (0, 1, 2)

POLICY_DEPENDENCE_NOTE = (
    "All absolute counts are reference-policy-dependent: a support set is the "
    "set of decisions one frozen candidate is driven through. Only the patterns "
    "are claimed (set equalities, the budget at which certification flips, and "
    "the maximum min-blunder distance). The reversed-tie-break row demonstrates "
    "the dependence directly."
)

CONJECTURE = {
    "status": ("unconditional form REFUTED at (4,4); survives only for "
               "SCALE-MATCHED grader/reference pairs. equality clause WITHDRAWN"),
    "statement": (
        "When the grader's scale matches the scale the reference policy "
        "optimizes, the opponent tremble budget is at most the number of value "
        "classes below the root value AS THE OPPONENT SEES IT -- the blundering "
        "side is the one with rungs left to fall through. Mismatch the two and "
        "the bound does not merely loosen, it FAILS: a candidate indifferent to "
        "rungs the grader is scoring hands value back, and the opponent "
        "re-descends a ladder it had already fallen down."
    ),
    "refutations": [
        "(4,4) seat 0, class grading against the WDL-optimal V1 reference: "
        "budget 3 from drawn roots where the bound is 2; budget 2 from roots "
        "where the bound is 1; and -- decisively -- budget 2 from roots the "
        "opponent has ALREADY LOST outright, where the bound is 0 and no "
        "further fall should be possible at all.",
        "The equality clause of the earlier form ('equality when the descent "
        "visits every intermediate class') fails in the other direction: at "
        "(3,3) seat 1 the class bound permits 3 at the 882 roots where the "
        "opponent is winning on attrition, and the observed budget is 1.",
    ],
    "what_survives": [
        "Scale-matched pairs are unviolated everywhere tested: WDL grading "
        "against V1, and class grading against the class-optimal V2, at both "
        "sizes and both seats -- exhaustively at (3,3) (all 10,882 and 15,228 "
        "candidate-to-move roots). The (4,4) evidence is weaker than the word "
        "'tested' suggests: the scan takes the 150 SHALLOWEST roots per value "
        "class within 8 ply, so the clean result there rests on a "
        "depth-biased subsample of 600 roots. A violation found in such a "
        "sample still refutes; a clean one does not confirm.",
        "The failures are exactly the mismatched pair, and they are the same "
        "mechanism as the (4,4) seat-0 certification cliff: V1's indifference "
        "between losing outright and losing on the tiebreaker. Section C showed "
        "that mismatch manufactures gating states; this shows it also breaks "
        "the budget bound.",
        "The mismatched pair is clean at (3,3) and refuted from (4,4) up -- the "
        "same size-dependence as Finding 8, and the same warning against "
        "reading a rule off the smallest board.",
    ],
    "why_not_a_law": (
        "What is left is an unproved upper bound with a scope condition found "
        "by testing it, exhaustive at one board size and sampled at one more. "
        "It is also loose where it would be informative: it forbids budgets "
        "nobody has observed rather than predicting the ones we do."
    ),
}


# ============================================================ grading scales

def zero_regret_class(state: GameState, memo) -> List[Move]:
    """5-class-ladder analogue of `ee.zero_regret_moves`, which grades on WDL.

    Moves are compared on the raw solver value (100/10/0/-10/-100) oriented to
    the mover, so trading a true win for an attrition win is a blunder here and
    is not one under WDL. Since WDL is a monotone collapse of the ladder, this
    set is always a subset of the WDL zero-regret set: a class grader is
    strictly stricter about what counts as a mistake.
    """
    moves = get_legal_moves(state)
    vals = {m: orient(memo[apply_move(state, m)], state.turn) for m in moves}
    best = max(vals.values())
    return [m for m in moves if vals[m] == best]


ZERO_REGRET: Dict[str, Callable[[GameState, Dict], List[Move]]] = {
    GRADE_WDL: ee.zero_regret_moves,
    GRADE_CLASS: zero_regret_class,
}


# ========================================================= reference policies

def canonical_optimal_v2(state: GameState, memo) -> Move:
    """CANONICAL_POLICY_V2: class-optimal first, `canonical_move_key` tie-break.

    A scale-matched reference: it optimizes the same 5-class ladder the class
    grader scores the opponent on. Still WDL-optimal (the ladder refines WDL),
    so `identifiability`'s best-compatible == game-value assertion holds.
    """
    moves = sorted(get_legal_moves(state), key=ee.canonical_move_key)
    vals = {m: orient(memo[apply_move(state, m)], state.turn) for m in moves}
    best = max(vals.values())
    for m in moves:
        if vals[m] == best:
            return m
    return moves[0]


def canonical_optimal_reversed(state: GameState, memo) -> Move:
    """CANONICAL_POLICY_V1_REVERSED: WDL-optimal, LAST under canonical_move_key.

    Exists solely as the robustness control. It is as arbitrary as V1 and no
    worse, so any pattern that survives the swap is not an artifact of the
    tie-break, and any count that moves under it was never a property of the
    game.
    """
    moves = sorted(get_legal_moves(state), key=ee.canonical_move_key, reverse=True)
    vals = {m: wdl(memo[apply_move(state, m)], state.turn) for m in moves}
    best = max(vals.values())
    for m in moves:
        if vals[m] == best:
            return m
    return moves[0]


REFERENCE: Dict[str, Callable[[GameState, Dict], Move]] = {
    POLICY_V1: ee.canonical_optimal,
    POLICY_V2: canonical_optimal_v2,
    POLICY_V1R: canonical_optimal_reversed,
}


# =============================================================== support sets

def tremble_support(memo, r: int, seat: int, budget: int,
                    zr_fn=ee.zero_regret_moves,
                    canon=ee.canonical_optimal,
                    root: Optional[GameState] = None) -> Set[GameState]:
    """Candidate decisions exposed when the opponent may blunder `budget` times.

    The candidate plays `canon` throughout; the opponent plays any zero-regret
    move for free and any other legal move at a cost of one, up to `budget` per
    game. `seen` is keyed on (state, spent) because reaching a state with budget
    to spare is not the same event as reaching it exhausted.

    budget=0 reproduces `ee.benchmark_support(..., "optimal")` exactly when
    zr_fn/canon are the defaults -- asserted as a wiring check, not offered as
    evidence.

    `root` defaults to the opening. It is parameterised for the conjecture scan,
    which needs roots whose value is not a draw; nothing else should pass it.
    """
    support: Set[GameState] = set()
    seen: Set[Tuple[GameState, int]] = set()
    stack: List[Tuple[GameState, int]] = [(root or empty_state(r, r, 0), 0)]
    while stack:
        s, b = stack.pop()
        if (s, b) in seen:
            continue
        seen.add((s, b))
        if ee.end_value(s) is not None:
            continue
        if s.turn == seat:
            support.add(s)
            stack.append((apply_move(s, canon(s, memo)), b))
        else:
            zr = set(zr_fn(s, memo))
            for m in zr:
                stack.append((apply_move(s, m), b))
            if b < budget:
                for m in get_legal_moves(s):
                    if m not in zr:
                        stack.append((apply_move(s, m), b + 1))
    return support


def support_all(memo, r: int, seat: int, canon=ee.canonical_optimal,
                root: Optional[GameState] = None) -> Set[GameState]:
    """Candidate decisions exposed against the unrestricted opponent universe.

    Equals `ee.benchmark_support(..., "all")` at canon=V1 from the opening;
    parameterised because the V1/V2 decomposition needs the same universe under
    a moved reference, and the conjecture scan needs it under a moved root.
    """
    support: Set[GameState] = set()
    seen: Set[GameState] = set()
    stack = [root or empty_state(r, r, 0)]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        if ee.end_value(s) is not None:
            continue
        if s.turn == seat:
            support.add(s)
            stack.append(apply_move(s, canon(s, memo)))
        else:
            for m in get_legal_moves(s):
                stack.append(apply_move(s, m))
    return support


# ======================================================== independent 0-1 BFS

def min_blunders(memo, r: int, seat: int, zr_fn=ee.zero_regret_moves,
                 canon=ee.canonical_optimal, track: bool = False,
                 root: Optional[GameState] = None
                 ) -> Tuple[Dict[GameState, int], Dict]:
    """Fewest opponent blunders needed to reach each state, by 0-1 BFS.

    A SEPARATE CODE PATH from `tremble_support` on purpose: it computes a
    distance, not a reachable set, so `{s : dist[s] <= m}` restricted to the
    candidate's turn reconstructing support(m) is a real cross-check of the
    budget bookkeeping rather than the same traversal run twice.

    With track=True the parent map is populated for value-trajectory
    reconstruction; it holds (predecessor, move, edge_weight) per state.
    """
    start = root or empty_state(r, r, 0)
    dist: Dict[GameState, int] = {start: 0}
    parent: Dict[GameState, Optional[Tuple]] = {start: None}
    dq = deque([start])
    while dq:
        s = dq.popleft()
        b = dist[s]
        if ee.end_value(s) is not None:
            continue
        if s.turn == seat:
            ns = apply_move(s, canon(s, memo))
            if ns not in dist or dist[ns] > b:
                dist[ns] = b
                if track:
                    parent[ns] = (s, None, 0)
                dq.appendleft(ns)
        else:
            zr = set(zr_fn(s, memo))
            for m in get_legal_moves(s):
                w = 0 if m in zr else 1
                ns = apply_move(s, m)
                if ns not in dist or dist[ns] > b + w:
                    dist[ns] = b + w
                    if track:
                        parent[ns] = (s, m, w)
                    (dq.appendleft if w == 0 else dq.append)(ns)
    return dist, parent


def blunder_path(state: GameState, parent: Dict) -> List[GameState]:
    """Root-to-`state` path along the BFS tree, root first."""
    path = [state]
    while parent.get(path[-1]) is not None:
        path.append(parent[path[-1]][0])
    return list(reversed(path))


# ==================================================================== solver

def compatible_extreme(memo, r: int, seat: int, constrained: Set[GameState],
                       canon=ee.canonical_optimal,
                       worst: bool = True) -> Tuple[int, int]:
    """`ee.compatible_extreme` generalized to an arbitrary reference policy.

    Identical semantics -- pinned inside `constrained`, adversary-driven or
    self-driven outside, opponent always minimizing, value decided before the
    depth tie-break -- with the pinning target passed in rather than hardwired
    to CANONICAL_POLICY_V1. `check_generalization_matches_ee` asserts the two
    agree exactly at canon=V1, so this stays a generalization and not a fork.

    The solve root stays hardwired to the opening even though the support
    builders now take one: certification is only ever claimed from the opening,
    and pinning a re-rooted support inside a solve that starts at the opening
    would answer a question nobody asked.
    """
    cache: Dict[GameState, Tuple[int, int]] = {}

    def solve(state: GameState) -> Tuple[int, int]:
        hit = cache.get(state)
        if hit is not None:
            return hit
        t = ee.end_value(state)
        if t is not None:
            out = (wdl(t, seat), 0)
            cache[state] = out
            return out

        own = state.turn == seat
        if own and state in constrained:
            opts, maximize = [canon(state, memo)], False
        elif own:
            opts, maximize = get_legal_moves(state), (not worst)
        else:
            opts, maximize = get_legal_moves(state), False

        best: Optional[Tuple[int, int]] = None
        for m in opts:
            v, d = solve(apply_move(state, m))
            cand = (v, d + 1)
            if best is None:
                best = cand
            elif maximize and (cand[0] > best[0]
                               or (cand[0] == best[0] and cand[1] < best[1])):
                best = cand
            elif not maximize and (cand[0] < best[0]
                                   or (cand[0] == best[0] and cand[1] < best[1])):
                best = cand
        cache[state] = best
        return best

    return solve(empty_state(r, r, 0))


def identifiability(memo, r: int, seat: int, C: Set[GameState],
                    canon=ee.canonical_optimal) -> Dict:
    """`ee.identifiability` against an arbitrary reference policy.

    Keeps the assertion that best-compatible equals the game value: every
    reference here is WDL-optimal, so a reference that failed it would be a
    solver bug, not a finding.
    """
    lo, dlo = compatible_extreme(memo, r, seat, C, canon, worst=True)
    hi, _ = compatible_extreme(memo, r, seat, C, canon, worst=False)
    game_value = wdl(memo[empty_state(r, r, 0)], seat)
    assert hi == game_value, (
        f"best-compatible {hi} != game value {game_value}; solver is wrong")
    return {"worst": ee.LBL[lo], "best": ee.LBL[hi], "identified": hi == lo,
            "shortest_certificate_depth_for_worst_value": dlo}


def check_generalization_matches_ee(memo, r: int, seat: int) -> bool:
    """Differential check: at canon=V1 this module's solver must agree with the
    committed one, value AND depth, on a support set that is known to leave a
    residual. Cheap; run wherever the size is cheap."""
    C = ee.benchmark_support(memo, r, seat, "optimal")
    for worst in (True, False):
        mine = compatible_extreme(memo, r, seat, C, ee.canonical_optimal, worst)
        theirs = ee.compatible_extreme(memo, r, seat, C, worst)
        assert mine == theirs, (
            f"generalized solver diverged from ee at ({r},{r}) seat {seat} "
            f"worst={worst}: {mine} != {theirs}")
    return True


# ============================================================ mechanism gates

def mechanism_gate(memo, r: int, seat: int, zr_fn=ee.zero_regret_moves,
                   canon=ee.canonical_optimal) -> Dict:
    """Why one blunder is enough, and where that reason breaks.

    Counts opponent-to-move states BELOW a blunder that are not fully
    zero-regret. Under WDL the count must be 0 -- once the opponent has thrown
    the game, every move it has left is zero-regret, so the rest of the line is
    free and the budget never needs a second unit.

    That holds here because the OPENING IS A DRAW at every size solved, putting
    the opponent one rung above the bottom, so its only available mistake lands
    on a loss. It is not a general fact about WDL grading: `conjecture_scan`
    re-rooted at states the opponent is winning finds budget 2, because a win
    can be thrown to a draw and the draw thrown again. This gate is called only
    from the opening, where the one-rung argument applies. Under class grading the count
    must be positive AND must be exactly the attrition-lost (-10) states: a
    player who is losing on the tiebreaker still has a ladder rung left to fall
    off, which is the whole mechanism behind the apparent second blunder.

    A gate asserting only the WDL half checks nothing: it would pass on any
    grading whose zero-regret sets happened to be total.
    """
    dist, _ = min_blunders(memo, r, seat, zr_fn, canon)
    below = violations = attrition_constrained = 0
    for s, b in dist.items():
        if b >= 1 and s.turn != seat and ee.end_value(s) is None:
            below += 1
            if len(zr_fn(s, memo)) != len(get_legal_moves(s)):
                violations += 1
                if orient(memo[s], 1 - seat) == ATTRITION_LOSS:
                    attrition_constrained += 1
    return {"below_blunder": below, "violations": violations,
            "attrition_constrained": attrition_constrained,
            "violations_are_exactly_attrition_lost":
                violations == attrition_constrained}


# =================================================================== one cell

def budget_cell(memo, r: int, seat: int, grading: str, policy: str,
                budgets: Sequence[int] = BUDGETS,
                certify: bool = True) -> Dict:
    """Everything the claim needs for one (size, seat, grading, reference) cell.

    Records support sizes and certification per budget, the all-legal support
    under the same reference, the independent-BFS min-blunder histogram over
    that support, and the ply depths of the states each extra unit of budget
    buys.
    """
    zr_fn, canon = ZERO_REGRET[grading], REFERENCE[policy]
    s_all = support_all(memo, r, seat, canon)

    rows: List[Dict] = []
    supports: Dict[int, Set[GameState]] = {}
    for m in budgets:
        sup = tremble_support(memo, r, seat, m, zr_fn, canon)
        supports[m] = sup
        row = {"budget": m, "pinned": len(sup), "equals_all_legal": sup == s_all}
        if certify:
            row.update(identifiability(memo, r, seat, sup, canon))
        rows.append(row)

    dist, _ = min_blunders(memo, r, seat, zr_fn, canon)
    hist = Counter(dist[s] for s in s_all)
    depths_by_budget = {
        str(b): sorted(ee.exact_ply_depth(s, r) for s in s_all if dist[s] == b)
        for b in sorted(hist)}

    # The cross-check: the distance labelling must rebuild each support exactly.
    bfs_rebuild = {
        str(m): {s for s in s_all if dist[s] <= m} == supports[m]
        for m in budgets}

    return {
        "grading": grading,
        "reference_policy": policy,
        "all_legal_pinned": len(s_all),
        "budgets": rows,
        "min_blunder_histogram": {str(k): v for k, v in sorted(hist.items())},
        "max_min_blunders": max(hist),
        "bfs_reconstructs_support": bfs_rebuild,
        "ply_depths": {
            b: {"count": len(d), "min": d[0], "median": d[len(d) // 2],
                "max": d[-1]}
            for b, d in depths_by_budget.items() if d},
        "certification_flips_at": (
            next((row["budget"] for row in rows if row.get("identified")), None)
            if certify else None),
    }


# ======================================================= the two-blunder family

def deepest_budget_family(memo, r: int, seat: int, grading: str = GRADE_CLASS,
                          policy: str = POLICY_V1) -> Dict:
    """The states only the LAST unit of budget reaches, described structurally.

    The count of these states is the wrong thing to report, and two plausible
    readings of it are both wrong:

      * not a symmetry artifact. `evaluation_equivalence` and this module run on
        the UN-FOLDED state space, so a single position with a trivial
        stabilizer would appear |D4| = 8 times over -- and the (3,3) family has
        exactly 8 members, which invites that reading. It is false: the 8 carry
        8 DISTINCT canonical keys (`tt_key`). Recorded as
        `distinct_up_to_symmetry` so the coincidence cannot be mistaken for a
        mechanism.
      * not eight independent facts. All 8 descend from ONE opponent decision
        node, and they are precisely that node's non-zero-regret children. The
        8 is a branching factor, and the honest statement is "the extra budget
        unit is demanded at a single decision".

    That node is also the one the mechanism gate counts: the lone opponent state
    below a blunder where the class grader still constrains the opponent.

    Value trajectories are oriented to the OPPONENT's seat, so a blunder is a
    step down and the reference's indifference between classes shows up as a
    step back up.
    """
    zr_fn, canon = ZERO_REGRET[grading], REFERENCE[policy]
    dist, parent = min_blunders(memo, r, seat, zr_fn, canon, track=True)
    s_all = support_all(memo, r, seat, canon)
    worst = max(dist[s] for s in s_all)
    family = [s for s in s_all if dist[s] == worst]

    # The decision nodes the last budget unit is spent at, and what they cost.
    nodes = sorted({parent[s][0] for s in family}, key=repr)
    node_rows = []
    covered: Set[GameState] = set()
    for n in nodes:
        legal, zr = get_legal_moves(n), zr_fn(n, memo)
        kids = {apply_move(n, m) for m in legal if m not in set(zr)}
        covered |= kids
        node_rows.append({
            "ply_depth": ee.exact_ply_depth(n, r),
            "min_blunders_to_node": dist[n],
            "value_opponent_seat": orient(memo[n], 1 - seat),
            "legal_moves": len(legal),
            "zero_regret_moves": len(zr),
            "blundering_children": len(kids),
        })

    rep = min(family, key=repr)
    return {
        "grading": grading,
        "reference_policy": policy,
        "budget": worst,
        "states_unfolded": len(family),
        "distinct_up_to_symmetry": len({tt_key(s) for s in family}),
        "is_one_d4_orbit": len({tt_key(s) for s in family}) == 1,
        "d4_order": 8,
        "decision_nodes": len(nodes),
        "family_is_exactly_the_blundering_children": covered == set(family),
        "nodes": node_rows,
        "reserves": sorted({s.res for s in family}),
        "removals_played": sorted({
            ee.exact_ply_depth(s, r) - (2 * r - sum(s.res)) for s in family}),
        "ply_depths": sorted({ee.exact_ply_depth(s, r) for s in family}),
        "value_trajectory_opponent_seat": [
            orient(memo[p], 1 - seat) for p in blunder_path(rep, parent)],
    }


# ========================================================= conjecture scan

# The 5-class ladder, best-first. WDL is the same ladder collapsed to three.
CLASS_LADDER: Tuple[int, ...] = (100, 10, 0, -10, -100)


def classes_below(value: int, grading: str) -> int:
    """How many rungs the OPPONENT still has to fall from a state worth `value`
    to it, under the grading in use.

    The conjecture is about the blundering side, so the count is taken from the
    opponent's seat: an opponent already at the bottom class cannot drop at all,
    and one that is winning outright has the whole ladder beneath it. The
    original statement said only "below the root value" -- this makes the seat
    explicit, which the drawn-root cells could not distinguish because a draw
    sits at the same rung from both seats.
    """
    ladder = CLASS_LADDER if grading == GRADE_CLASS else (1, 0, -1)
    v = value if grading == GRADE_CLASS else (1 if value > 0 else -1 if value < 0 else 0)
    return sum(1 for c in ladder if c < v)


def is_scale_matched(grading: str, policy: str) -> bool:
    """Is the reference policy optimal on the scale the grader is scoring?

    V1 and its reversed twin optimize WDL only; V2 optimizes the 5-class ladder
    and is therefore WDL-optimal too, since WDL is that ladder collapsed. So the
    class grader is matched by V2 alone -- and that one mismatched combination
    is where both the certification cliff and the bound failure live.
    """
    return grading != GRADE_CLASS or policy == POLICY_V2


def budget_from_root(memo, r: int, seat: int, root: GameState,
                     grading: str, policy: str = POLICY_V1) -> int:
    """Max min-blunders over the all-legal support reached from `root`."""
    zr_fn, canon = ZERO_REGRET[grading], REFERENCE[policy]
    dist, _ = min_blunders(memo, r, seat, zr_fn, canon, root=root)
    sup = support_all(memo, r, seat, canon, root=root)
    return max((dist[s] for s in sup), default=0)


def conjecture_scan(memo, r: int, seat: int, grading: str,
                    max_depth: Optional[int] = None,
                    per_class: Optional[int] = None,
                    policy: str = POLICY_V1) -> Dict:
    """Does the class-count bound survive roots that are not drawn?

    EVERY cell in the main experiment starts from the opening, which is a draw
    at every size solved here. So "classes below the root value" is 2 under
    class grading and 1 under WDL in all of them, and the observed maxima (2 and
    1) sit exactly at the bound -- consistent with the conjecture, but equally
    consistent with the budget being capped for reasons having nothing to do
    with class counts. The two hypotheses are indistinguishable at a drawn root.

    This scan separates them. It re-roots the same machinery at states whose
    value is NOT a draw, where the bound takes values 0 through 4, and asks
    whether the observed budget respects it and whether it is ever attained. A
    root where the opponent is winning outright has four rungs beneath it; if
    the budget there is still capped at 2, the class-count framing is wrong and
    the conjecture should be withdrawn rather than reworded.

    Roots are non-terminal candidate-to-move states (the same turn convention as
    the opening). Both limits default to off: at (3,3) EVERY such state is
    tested, so the result is a statement about the whole game and not a sample.
    `max_depth` and `per_class` exist for larger boards, and cap by a
    deterministic order so the scan cannot drift with hashing.

    That order is (ply depth, repr), so capping keeps the SHALLOWEST roots in
    each class -- reproducible, but biased, and biased in the direction that
    matters: shallow roots have the most game left and so the most room to
    exceed the bound. Treat a violation from a capped scan as conclusive and a
    clean capped scan as suggestive only. `exhaustive` in the result records
    which of the two a reader is looking at.
    """
    by_class: Dict[int, List[GameState]] = {}
    for s in memo:
        if s.turn != seat or ee.end_value(s) is not None:
            continue
        if max_depth is not None and ee.exact_ply_depth(s, r) > max_depth:
            continue
        by_class.setdefault(orient(memo[s], 1 - seat), []).append(s)

    rows: List[Dict] = []
    for value in CLASS_LADDER:
        pool = sorted(by_class.get(value, []),
                      key=lambda s: (ee.exact_ply_depth(s, r), repr(s)))
        if not pool:
            continue
        if per_class is not None:
            pool = pool[:per_class]
        bound = classes_below(value, grading)
        observed = [budget_from_root(memo, r, seat, s, grading, policy) for s in pool]
        rows.append({
            "root_value_opponent_seat": value,
            "classes_below": bound,
            "roots_tested": len(pool),
            "max_budget_observed": max(observed),
            "attains_bound": max(observed) == bound,
            "exceeds_bound": max(observed) > bound,
            "budget_histogram": {str(k): v for k, v in sorted(Counter(observed).items())},
        })

    return {
        "grading": grading,
        "reference_policy": policy,
        "scale_matched": is_scale_matched(grading, policy),
        "max_root_depth": max_depth,
        "roots_per_class": per_class,
        "exhaustive": max_depth is None and per_class is None,
        "roots_tested": sum(row["roots_tested"] for row in rows),
        "rows": rows,
        "bound_holds_everywhere": not any(row["exceeds_bound"] for row in rows),
        "bound_attained_somewhere": [row["root_value_opponent_seat"]
                                     for row in rows if row["attains_bound"]],
        "budgets_above_two_observed": max(
            (row["max_budget_observed"] for row in rows), default=0) > 2,
    }


# ================================================================ per size

def gate_c_agreement(memo, r: int, seat: int) -> Dict:
    """Cheapest available validation: support(0) and the all-legal support must
    be SET-equal to Gate C's optimal-only and all-legal supports.

    The first is close to a tautology (budget 0 IS the optimal-only opponent);
    it is here to catch a wiring error, not to support the claim.
    """
    opt = ee.benchmark_support(memo, r, seat, "optimal")
    alle = ee.benchmark_support(memo, r, seat, "all")
    return {
        "support0_equals_gate_c_optimal":
            tremble_support(memo, r, seat, 0) == opt,
        "support_all_equals_gate_c_all": support_all(memo, r, seat) == alle,
        "gate_c_optimal_pinned": len(opt),
        "gate_c_all_pinned": len(alle),
        "caveat": ("support(0) == optimal-only is a wiring check: budget 0 is "
                   "the optimal-only opponent by construction."),
    }


def run_size(r: int, seats: Sequence[int] = (0, 1), full: bool = True,
             certify: bool = True) -> Dict:
    """Every cell for one board size, from a single solve.

    full=False keeps only the WDL/V1 headline; certify=False drops the pinned
    solves. Both are for (5,5), where the state space makes the class-ladder
    cross product pointless and `compatible_extreme` memory-bound -- the same
    wall that kept Finding 17's Gate C at (4,4) while Gate A reached (5,5).
    """
    t0 = time.time()
    memo = solve_all(empty_state(r, r, 0))
    print(f"\n=== ({r},{r}): {len(memo):,} states [{time.time()-t0:.0f}s solve] ===")

    rec: Dict = {"states": len(memo), "certified": certify, "seats": {}}
    for seat in seats:
        if certify:
            check_generalization_matches_ee(memo, r, seat)
        row: Dict = {"gate_c_agreement": gate_c_agreement(memo, r, seat)}

        row["wdl_v1"] = budget_cell(memo, r, seat, GRADE_WDL, POLICY_V1,
                                    certify=certify)
        row["mechanism_wdl"] = mechanism_gate(
            memo, r, seat, ee.zero_regret_moves, ee.canonical_optimal)
        _print_cell(r, seat, "WDL / V1", row["wdl_v1"], row["mechanism_wdl"])

        if full:
            row["class_v1"] = budget_cell(memo, r, seat, GRADE_CLASS, POLICY_V1,
                                          certify=certify)
            row["mechanism_class"] = mechanism_gate(
                memo, r, seat, zero_regret_class, ee.canonical_optimal)
            _print_cell(r, seat, "class / V1", row["class_v1"],
                        row["mechanism_class"])

            row["class_v2"] = budget_cell(memo, r, seat, GRADE_CLASS, POLICY_V2,
                                          certify=certify)
            row["mechanism_class_v2"] = mechanism_gate(
                memo, r, seat, zero_regret_class, canonical_optimal_v2)
            _print_cell(r, seat, "class / V2", row["class_v2"],
                        row["mechanism_class_v2"])

            row["wdl_v1_reversed"] = budget_cell(
                memo, r, seat, GRADE_WDL, POLICY_V1R, certify=certify)
            _print_cell(r, seat, "WDL / V1-reversed", row["wdl_v1_reversed"])

            # The conjecture's own falsification attempt, over the two matched
            # grader/reference pairs and the one mismatched pair. Exhaustive at
            # (3,3); capped above it, where the root set is six figures.
            depth, cap = (None, None) if r <= 3 else (8, 150)
            row["conjecture_scan"] = {
                f"{g}_{p[17:].lower()}": conjecture_scan(memo, r, seat, g, depth, cap, p)
                for g, p in ((GRADE_WDL, POLICY_V1), (GRADE_CLASS, POLICY_V1),
                             (GRADE_CLASS, POLICY_V2))}
            for label, sc in row["conjecture_scan"].items():
                bad = [(row_["root_value_opponent_seat"], row_["classes_below"],
                        row_["max_budget_observed"])
                       for row_ in sc["rows"] if row_["exceeds_bound"]]
                print(f"    conjecture scan [{label:<10} "
                      f"{'matched' if sc['scale_matched'] else 'MISMATCHED'}]: "
                      f"{sc['roots_tested']:,} roots"
                      f"{' exhaustive' if sc['exhaustive'] else ''} | "
                      f"bound holds={sc['bound_holds_everywhere']}"
                      + (f" | VIOLATIONS (value,bound,observed)={bad}" if bad else ""))

            if row["class_v1"]["max_min_blunders"] >= 2:
                fam = deepest_budget_family(memo, r, seat)
                row["two_blunder_family"] = fam
                print(f"    two-blunder family: {fam['states_unfolded']} states "
                      f"= {fam['distinct_up_to_symmetry']} up to symmetry, from "
                      f"{fam['decision_nodes']} decision node(s), reserves "
                      f"{fam['reserves']}, removals {fam['removals_played']}, "
                      f"trajectory {fam['value_trajectory_opponent_seat']}")

        rec["seats"][str(seat)] = row
    del memo
    return rec


def _print_cell(r: int, seat: int, label: str, cell: Dict,
                mech: Optional[Dict] = None) -> None:
    line = " ".join(
        f"m={row['budget']}:{row['pinned']}"
        f"{'=all' if row['equals_all_legal'] else ''}"
        f"/{row.get('worst', '?')}" for row in cell["budgets"])
    print(f"  ({r},{r}) seat {seat} {label:<18} S_all={cell['all_legal_pinned']:>5,} "
          f"| {line} | max_min_blunders={cell['max_min_blunders']} "
          f"| hist={cell['min_blunder_histogram']}")
    if mech is not None:
        print(f"    mechanism: below_blunder={mech['below_blunder']} "
              f"violations={mech['violations']} "
              f"attrition_constrained={mech['attrition_constrained']}")


# ================================================================== driver

def main(sizes: Tuple[int, ...] = (3,), slow: bool = False,
         five: bool = False) -> None:
    t0 = time.time()
    if slow and 4 not in sizes:
        sizes = tuple(sizes) + (4,)

    payload: Dict = {
        "note": POLICY_DEPENDENCE_NOTE,
        "conjecture": CONJECTURE,
        "convention": {
            "certification": "WDL (compatible_extreme, unchanged from Finding 17)",
            "gradings": {GRADE_WDL: "ee.zero_regret_moves",
                         GRADE_CLASS: "raw 5-class ladder 100/10/0/-10/-100"},
            "reference_policies": {
                POLICY_V1: "WDL-optimal, canonical_move_key tie-break",
                POLICY_V2: "class-optimal, canonical_move_key tie-break",
                POLICY_V1R: "WDL-optimal, reversed canonical_move_key tie-break"},
            "canonical_order": ee.CANONICAL_ORDER_DEFINITION,
            "budgets": list(BUDGETS),
        },
        "sizes": {},
    }
    for r in sizes:
        payload["sizes"][f"({r},{r})"] = run_size(r)

    if five:
        # (5,5) carries the headline only, and without certification: at 12.7M
        # states the pinned solves hit the same memory wall that kept Gate C at
        # (4,4). What survives is the support claim -- one blunder reaches the
        # whole all-legal support -- which is the half a reviewer asks about.
        payload["headline_5_5"] = run_size(5, full=False, certify=False)

    config = {"sizes": list(sizes), "slow": slow, "five": five,
              "budgets": list(BUDGETS),
              "reference_policy_default": POLICY_V1,
              "counts_are_reference_policy_dependent": True}
    path = write_result(NAME, config, payload)
    announce(NAME, path)
    print(f"\n[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    main(tuple(int(a) for a in args) if args else (3,),
         slow="--slow" in flags, five="--tb5" in flags)
