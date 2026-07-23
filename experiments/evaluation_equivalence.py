"""Residual worst-case uncertainty: what does passing an evaluation RULE OUT?

    After an agent passes an evaluation, how bad can a policy still be while
    remaining consistent with everything that evaluation observed?

A policy is BENCHMARK-COMPATIBLE if it agrees with the reference wherever the
evaluation constrained it (set C). Over all compatible policies we report the
COMPATIBLE OUTCOME RANGE as an ordered pair (worst, best) -- the evaluation's
certification power.

Computed EXACTLY via best-response with PINNED NODES, not policy search:
in C the mover is forced to the reference action; outside C the adversary picks
the mover's action (worst case) or the mover picks its own (best case).

====================================================================
OBSERVATION LEVELS -- an evaluator is defined by which states it visits AND
what it observes there. Three rungs, decreasing information:

  transcript  observes and constrains the ACTION.        <-- implemented here
  grade       observes only that the action was zero-regret.  NOT IMPLEMENTED
  outcome     observes only terminal results (win rate, Elo). NOT IMPLEMENTED

The compatible policy class GROWS at each rung. Mixing rungs in one table
without labelling is the single easiest way to produce a wrong result.
====================================================================

*** grade / regret0 IS UNSOUND AS A ONE-PASS PINNED SOLVE ***
If the candidate may play ANY zero-regret action, deviating within C sends the
game to states C never covered, where a one-pass solver lets it misbehave --
but a real evaluation would have FOLLOWED it there. Verified (3,3) seat 1:
twin deviates ply 1, escapes support ply 3, exploits an unobserved state,
yielding a spurious "forced loss survives full support". The correct form is a
fixpoint over the policy's OWN reachable support.

*** outcome-only (win rate, Elo) MUST NOT be modelled by pinning actions ***
A policy may deviate from canonical and still post identical win/loss records.
Pinning encountered actions grants the evaluator information it never had and
overstates its certification power.
"""

import hashlib
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from collapse3.enumeration import solve_all, wdl
from collapse3.game import (GameState, Move, apply_move, attrition_value,
                            empty_state, evaluate_terminal, get_legal_moves)
from experiments._provenance import announce, write_result

NAME = "evaluation_equivalence"

CANONICAL_POLICY_VERSION = "CANONICAL_POLICY_V1"
CANONICAL_ORDER_DEFINITION = "(action_rank, peg, index); place=0, remove=1"
LBL = {1: "win", 0: "draw", -1: "LOSS"}

# ---- Horizon convention (see design doc section 10) -------------------------
# "Coverage through N plies" pins every candidate decision whose state ply-depth
# (the exact invariant, exact_ply_depth) is <= N. This is INCLUSIVE: a candidate
# decision occurring exactly at ply N is observed. The alternative (strict, < N)
# is available via the `inclusive` flag. The convention is stamped into results
# so any depth-limited number is unambiguous; the provisional pre-freeze figures
# (depth-4 = 10/62, depth-6 = 71/485) are NOT inherited -- they are recomputed
# under this convention wherever quoted.
HORIZON_INCLUSIVE = True

# ---- Gate 0 anchors (reproduce-or-abort) ------------------------------------
# Independently reproduced on this engine (commit at first freeze). Each row:
# global decision count, frozen-policy hash, control (all-pinned) outcome, and
# Gate C (optimal-only / all-legal) as (pinned_count, worst_outcome).
GATE0_ANCHORS: Dict[Tuple[int, int], Dict[int, Dict]] = {
    (2, 2): {
        0: {"global": 172, "hash": "ffc204f095eb9150", "control": "draw",
            "optimal": (9, "draw"), "all": (19, "draw")},
        1: {"global": 504, "hash": "4b404e24874c89aa", "control": "draw",
            "optimal": (62, "draw"), "all": (62, "draw")},
    },
    (3, 3): {
        0: {"global": 10882, "hash": "b9204bd62ed5e8cb", "control": "draw",
            "optimal": (41, "LOSS"), "all": (96, "draw")},
        1: {"global": 15228, "hash": "f4f8e9548e31e5b5", "control": "draw",
            "optimal": (367, "draw"), "all": (538, "draw")},
    },
}


def end_value(state: GameState) -> Optional[int]:
    t = evaluate_terminal(state)
    if t is not None:
        return t
    if not get_legal_moves(state):
        return attrition_value(state.board)
    return None


# =============================================================== ply depth

def exact_ply_depth(state: GameState, init_res: int) -> int:
    """EXACT plies from the root, closed form:

        plies = 2 * (beads placed) - (beads on board)

    Each placement is one ply moving a bead reserve->board; each removal is one
    ply destroying a board bead, so removals = placed - on_board.

    Validated against BFS over all 97,093 (3,3) states: exact everywhere, and
    NO state is reachable at two different depths (depth is a state invariant).

    DO NOT substitute material counts. `board_beads + reserves` is INVARIANT
    under placement (it reads 6 at every ply of the (3,3) opening) and is not a
    depth proxy -- an earlier draft used it and silently voided the control it
    existed to provide.
    """
    placed = 2 * init_res - (state.res[0] + state.res[1])
    return 2 * placed - sum(len(p) for p in state.board)


def verify_depth_formula(init_res: int) -> Tuple[int, int]:
    """BFS the reachable graph and check the closed form.
    Returns (states_checked, mismatches); mismatches MUST be 0."""
    root = empty_state(init_res, init_res, 0)
    depth = {root: 0}
    q = deque([root])
    while q:
        s = q.popleft()
        if end_value(s) is not None:
            continue
        for m in get_legal_moves(s):
            c = apply_move(s, m)
            if c not in depth:
                depth[c] = depth[s] + 1
                q.append(c)
    bad = sum(1 for s, d in depth.items() if exact_ply_depth(s, init_res) != d)
    return len(depth), bad


# ======================================================= canonical policy

ACTION_ORDER = {"place": 0, "remove": 1}


def canonical_move_key(m: Move):
    """Stable key from GAME SEMANTICS, never from str()/object representation.
    Move layout verified against the engine: ('place', peg) | ('remove', peg, idx)."""
    rank = ACTION_ORDER[m[0]]
    peg = m[1] if len(m) > 1 else -1
    idx = m[2] if len(m) > 2 else -1
    return (rank, peg, idx)


def canonical_optimal(state: GameState, memo) -> Move:
    """CANONICAL_POLICY_V1: the value-optimal move that is first under
    canonical_move_key. Frozen so refactors of legal-move ordering cannot
    silently change the strict equivalence class."""
    moves = sorted(get_legal_moves(state), key=canonical_move_key)
    vals = {m: wdl(memo[apply_move(state, m)], state.turn) for m in moves}
    best = max(vals.values())
    for m in moves:
        if vals[m] == best:
            return m
    return moves[0]


def canonical_policy_hash(memo, seat: int) -> str:
    """Regression anchor for the frozen policy. Expensive at (4,4); call in
    tests at (2,2)/(3,3) and stamp into results."""
    h = hashlib.sha256()
    states = [s for s in memo if end_value(s) is None and s.turn == seat]
    for s in sorted(states, key=repr):
        h.update(repr((s, canonical_optimal(s, memo))).encode())
    return h.hexdigest()[:16]


def zero_regret_moves(state: GameState, memo) -> List[Move]:
    moves = get_legal_moves(state)
    vals = {m: wdl(memo[apply_move(state, m)], state.turn) for m in moves}
    best = max(vals.values())
    return [m for m in moves if vals[m] == best]


# ================================================== quarantined protocols

def grade_level_residual(*_a, **_k):
    raise NotImplementedError(
        "grade/regret0 requires a self-consistent support fixpoint; a one-pass "
        "pinned-node solve is unsound (see module docstring)."
    )


def outcome_level_residual(*_a, **_k):
    raise NotImplementedError(
        "outcome-only evaluators (win rate, Elo) observe terminal results, not "
        "actions. Pinning encountered actions overstates certification power. "
        "Implement as a distinct constrained-line protocol."
    )


# ================================================================= solver

def compatible_extreme(memo, r: int, seat: int, constrained: Set[GameState],
                       worst: bool = True) -> Tuple[int, int]:
    """Exact extreme over TRANSCRIPT-compatible policies.

    in `constrained`     -> pinned to CANONICAL_POLICY_V1 (self-consistent)
    outside, worst=True  -> adversary picks the candidate's move  (residual)
    outside, worst=False -> candidate picks its own best move     (best compatible)
    opponent seat        -> always minimizes the candidate's outcome

    Returns (wdl_from_seat, shortest_depth_achieving_that_value).
    Depth is a REPORTING tie-break only: values are decided before it, so depth
    never affects the value. It is not rational play -- a doomed player might
    rationally maximize time-to-loss.
    """
    cache: Dict[GameState, Tuple[int, int]] = {}
    sys.setrecursionlimit(1_000_000)

    def solve(state: GameState) -> Tuple[int, int]:
        hit = cache.get(state)
        if hit is not None:
            return hit
        t = end_value(state)
        if t is not None:
            out = (wdl(t, seat), 0)
            cache[state] = out
            return out

        own = state.turn == seat
        if own and state in constrained:
            opts, maximize = [canonical_optimal(state, memo)], False
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


def identifiability(memo, r: int, seat: int, C: Set[GameState]) -> Dict:
    """Compatible outcome RANGE as an ordered pair. No scalar width: WDL and the
    outcome ladder are ordinal, so a numeric difference implies cardinal meaning
    the scale does not carry."""
    lo, dlo = compatible_extreme(memo, r, seat, C, worst=True)
    hi, _ = compatible_extreme(memo, r, seat, C, worst=False)
    game_value = wdl(memo[empty_state(r, r, 0)], seat)
    # ASSERTION, not a finding: the reference is compatible with its own
    # observations and is optimal, so best-compatible must equal the game value.
    assert hi == game_value, (
        f"best-compatible {hi} != game value {game_value}; solver is wrong")
    return {"worst": LBL[lo],
            "shortest_certificate_depth_for_worst_value": dlo,
            "best": LBL[hi],
            "identified": hi == lo}


# =========================================================== support sets

def benchmark_support(memo, r: int, seat: int, opponents: str) -> Set[GameState]:
    """EXECUTION support: every candidate decision the benchmark can ever expose.
    Candidate plays canonical; opponent family 'optimal' (value-optimal moves
    only) or 'all' (any legal move).

    If a protocol reaches a state without inspecting the candidate's action
    there, pin only the OBSERVATION subset -- execution != observation.
    Self-consistent for transcript compatibility only.
    """
    support: Set[GameState] = set()
    seen: Set[GameState] = set()
    stack = [empty_state(r, r, 0)]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        if end_value(s) is not None:
            continue
        if s.turn == seat:
            support.add(s)
            stack.append(apply_move(s, canonical_optimal(s, memo)))
        else:
            nxt = get_legal_moves(s) if opponents == "all" else zero_regret_moves(s, memo)
            for m in nxt:
                stack.append(apply_move(s, m))
    return support


def adversarial_support(memo, r: int, seat: int, max_plies: int,
                        inclusive: bool = HORIZON_INCLUSIVE) -> Set[GameState]:
    """Candidate decisions an arbitrary opponent can force within max_plies.

    Horizon convention (design doc section 10), made explicit and using the
    exact ply-depth invariant so it cannot drift with traversal order:

      inclusive=True  -> pin candidate decisions at state ply-depth <= max_plies
      inclusive=False -> pin candidate decisions at state ply-depth <  max_plies

    Either way, traversal stops expanding beyond max_plies. Because ply-depth is
    a state invariant (verify_depth_formula), a plain per-state `seen` set is
    sound -- no (state, ply) pairs needed.
    """
    support: Set[GameState] = set()
    seen: Set[GameState] = set()
    stack = [empty_state(r, r, 0)]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        if end_value(s) is not None:
            continue
        d = exact_ply_depth(s, r)
        if d > max_plies:                         # beyond the horizon: do not expand
            continue
        if s.turn == seat:
            if d <= max_plies if inclusive else d < max_plies:
                support.add(s)
            stack.append(apply_move(s, canonical_optimal(s, memo)))
        else:
            for m in get_legal_moves(s):
                stack.append(apply_move(s, m))
    return support


def sampled_support(memo, r: int, seat: int, games: int, seed: int,
                    noise: float = 0.25) -> Set[GameState]:
    """States constrained by `games` transcript-audited benchmark games."""
    import random
    rng = random.Random(seed)
    support: Set[GameState] = set()
    for _ in range(games):
        s = empty_state(r, r, 0)
        while end_value(s) is None:
            if s.turn == seat:
                support.add(s)
                s = apply_move(s, canonical_optimal(s, memo))
            else:
                mv = get_legal_moves(s)
                s = apply_move(s, rng.choice(mv) if rng.random() < noise
                               else rng.choice(zero_regret_moves(s, memo)))
    return support


def depth_matched_random(memo, r: int, seat: int, target: Set[GameState],
                         seed: int, universe: Optional[Set[GameState]] = None
                         ) -> Optional[Set[GameState]]:
    """Random state set matched to `target`'s EXACT ply-depth histogram.

    universe=None -> draw from all candidate decision states       (control A)
    universe=set  -> draw from that set, e.g. full exposure support (control B,
                     the strong one: same universe, same depths, only the
                     selection rule differs)

    Returns None if the universe cannot supply the histogram. Callers MUST
    abort the comparison rather than silently mismatch.
    """
    import random
    rng = random.Random(seed)
    pool_src = universe if universe is not None else {
        s for s in memo if end_value(s) is None and s.turn == seat}
    by_depth = defaultdict(list)
    for s in pool_src:
        by_depth[exact_ply_depth(s, r)].append(s)
    want = Counter(exact_ply_depth(s, r) for s in target)
    out: Set[GameState] = set()
    for d, k in want.items():
        pool = list(by_depth.get(d, []))
        if len(pool) < k:
            return None
        rng.shuffle(pool)
        out.update(pool[:k])
    return out


# ================================================================ Gate C

def gate_c_seat(memo, r: int, seat: int, want_hash: bool = True) -> Dict:
    """Transcript-level Gate C for one seat: global/exposure denominators, the
    frozen-policy hash, the control (all-pinned) outcome, and the optimal-only
    vs all-legal benchmark supports with their compatible outcome ranges."""
    glob = {s for s in memo if end_value(s) is None and s.turn == seat}
    expo = benchmark_support(memo, r, seat, "all")
    row: Dict = {
        "global_decisions": len(glob),
        "exposure": len(expo),
        "policy_hash": canonical_policy_hash(memo, seat) if want_hash else None,
        "control_all_pinned": identifiability(memo, r, seat, glob),
        "benchmark": {},
    }
    for opps in ("optimal", "all"):
        sup = benchmark_support(memo, r, seat, opps)
        idl = identifiability(memo, r, seat, sup)
        row["benchmark"][opps] = {
            "pinned": len(sup),
            "global_pct": round(100 * len(sup) / len(glob), 3),
            "exposure_pct": round(100 * len(sup) / len(expo), 3),
            **idl,
        }
    return row


# =============================================================== reproduce gate

def reproduce_gate0() -> Dict:
    """Reproduce-or-abort. Recompute the (2,2) and (3,3) anchors and compare to
    GATE0_ANCHORS exactly. Raises AssertionError on any mismatch so no headline
    experiment can run against a drifted engine or a changed canonical policy."""
    record: Dict = {"convention": {"policy": CANONICAL_POLICY_VERSION,
                                   "order": CANONICAL_ORDER_DEFINITION,
                                   "horizon_inclusive": HORIZON_INCLUSIVE},
                    "sizes": {}}
    for (r, _), seats in GATE0_ANCHORS.items():
        n, bad = verify_depth_formula(r)
        assert bad == 0, f"({r},{r}) ply-depth formula failed on {bad}/{n} states"
        memo = solve_all(empty_state(r, r, 0))
        size_rec: Dict = {"depth_states": n}
        for seat, anc in seats.items():
            row = gate_c_seat(memo, r, seat)
            got_hash = row["policy_hash"]
            got_control = row["control_all_pinned"]["worst"]
            got = {
                "global": row["global_decisions"],
                "hash": got_hash,
                "control": got_control,
                "optimal": (row["benchmark"]["optimal"]["pinned"],
                            row["benchmark"]["optimal"]["worst"]),
                "all": (row["benchmark"]["all"]["pinned"],
                        row["benchmark"]["all"]["worst"]),
            }
            for key in ("global", "hash", "control", "optimal", "all"):
                assert got[key] == anc[key], (
                    f"Gate 0 MISMATCH ({r},{r}) seat {seat} [{key}]: "
                    f"expected {anc[key]!r}, got {got[key]!r}")
            size_rec[f"seat{seat}"] = row
        record["sizes"][f"({r},{r})"] = size_rec
    return record


# ================================================================ driver

def main(sizes: Tuple[int, ...] = (3,), full: bool = False,
         control_seeds: int = 30, full_sizes: Tuple[int, ...] = (3,)) -> None:
    t0 = time.time()

    print("=== Gate 0: reproduce-or-abort (2,2)+(3,3) anchors ===")
    gate0 = reproduce_gate0()
    print(f"Gate 0 PASSED  [{time.time()-t0:.0f}s]  "
          f"policy={CANONICAL_POLICY_VERSION} horizon_inclusive={HORIZON_INCLUSIVE}")

    gate_c: Dict = {}
    for r in sizes:
        tr = time.time()
        memo = solve_all(empty_state(r, r, 0))
        print(f"\n=== Gate C ({r},{r}): {len(memo):,} states "
              f"[{time.time()-tr:.0f}s solve] ===")
        # Policy hash is expensive at (4,4)+; only recompute it where it's cheap.
        want_hash = r <= 3
        seats: Dict = {}
        for seat in (0, 1):
            row = gate_c_seat(memo, r, seat, want_hash=want_hash)
            seats[str(seat)] = row
            o, a = row["benchmark"]["optimal"], row["benchmark"]["all"]
            print(f"  seat {seat} | global={row['global_decisions']:,} "
                  f"exposure={row['exposure']:,}")
            print(f"    control all-pinned: worst={row['control_all_pinned']['worst']}")
            print(f"    vs optimal-only pinned={o['pinned']:>6,} "
                  f"({o['global_pct']:.1f}% global / {o['exposure_pct']:.1f}% exposure) "
                  f"-> worst={o['worst']:<4} identified={o['identified']}")
            print(f"    vs all-legal    pinned={a['pinned']:>6,} "
                  f"({a['global_pct']:.1f}% global / {a['exposure_pct']:.1f}% exposure) "
                  f"-> worst={a['worst']:<4} identified={a['identified']}")
        gate_c[f"({r},{r})"] = seats

    payload: Dict = {"gate0": gate0, "gate_c": gate_c}

    if full:
        # Gate D/B run at their own (feasible) sizes -- Gate C may span larger
        # boards where the D/B sweeps are intractable.
        payload["gate_d"] = _gate_d(full_sizes, control_seeds)
        payload["gate_b"] = _gate_b(full_sizes)

    config = {"sizes": list(sizes), "full": full, "full_sizes": list(full_sizes),
              "control_seeds": control_seeds if full else None,
              "canonical_policy_version": CANONICAL_POLICY_VERSION,
              "canonical_order": CANONICAL_ORDER_DEFINITION,
              "horizon_inclusive": HORIZON_INCLUSIVE}
    path = write_result(NAME, config, payload)
    announce(NAME, path)
    print(f"\n[total {time.time()-t0:.0f}s]")


def _gate_d(sizes: Tuple[int, ...], control_seeds: int) -> Dict:
    """Gate D: strategic depth-limited coverage vs two depth-matched controls.
    H2 is currently falsified at these budgets -- report plainly either way."""
    out: Dict = {}
    for r in sizes:
        memo = solve_all(empty_state(r, r, 0))
        size_rec: Dict = {}
        for seat in (0, 1):
            expo = benchmark_support(memo, r, seat, "all")
            rows = []
            for plies in (2, 4, 6, 8):
                sup = adversarial_support(memo, r, seat, plies)
                idl = identifiability(memo, r, seat, sup)
                controls = {}
                for label, uni in (("all_states", None), ("in_support", expo)):
                    losses = skipped = 0
                    for sd in range(control_seeds):
                        ctl = depth_matched_random(memo, r, seat, sup, sd, uni)
                        if ctl is None:
                            skipped += 1
                            continue
                        v, _ = compatible_extreme(memo, r, seat, ctl, worst=True)
                        losses += (v < 0)
                    controls[label] = {"force_losable": losses,
                                       "evaluated": control_seeds - skipped,
                                       "unmatchable": skipped}
                rows.append({"plies": plies, "pinned": len(sup),
                             "strategic_worst": idl["worst"], "controls": controls})
            size_rec[str(seat)] = rows
        out[f"({r},{r})"] = size_rec
    return out


def _gate_b(sizes: Tuple[int, ...]) -> Dict:
    """Gate B: transcript-audited budget curve."""
    out: Dict = {}
    for r in sizes:
        memo = solve_all(empty_state(r, r, 0))
        size_rec: Dict = {}
        for seat in (0, 1):
            glob = {s for s in memo if end_value(s) is None and s.turn == seat}
            expo = benchmark_support(memo, r, seat, "all")
            rows = []
            for games in (10, 30, 100, 300, 1000, 3000):
                sup = sampled_support(memo, r, seat, games, seed=0)
                idl = identifiability(memo, r, seat, sup)
                rows.append({"games": games, "pinned": len(sup),
                             "global_pct": round(100 * len(sup) / len(glob), 3),
                             "exposure_pct": round(100 * len(sup) / len(expo), 3),
                             **idl})
            size_rec[str(seat)] = rows
        out[f"({r},{r})"] = size_rec
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    sizes = tuple(int(a) for a in args) if args else (3,)
    main(sizes, full=("--full" in flags))
