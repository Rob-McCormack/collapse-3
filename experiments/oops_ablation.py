"""Is the Oops rule load-bearing? An exact ablation.

The Oops rule (``rules.md``): if your Removal's gravity cascade completes a line
for your *opponent alone*, the opponent wins immediately. It is the game's most
distinctive trap. Does it actually matter?

Split the question in two, because they have different answers:

  Q1 (values)    Does the Oops rule change what perfect play can achieve?
  Q2 (behaviour) Does perfect play ever *choose* an Oops?

There is a short argument that constrains both, which this experiment states and
then verifies exhaustively rather than assuming:

  LEMMA 1.  A genuine Oops hands the opponent a line, so its value to the mover
            is exactly TRUE_LOSS -- the bottom of the outcome ladder. (If the
            cascade lines up *both* players, the Simultaneous rule awards the
            mover the win, so that is not an Oops.)
  LEMMA 2.  Hence an Oops is never strictly better than any alternative. It can
            be optimal only in a position whose value is already TRUE_LOSS, in
            which case it is *tied* with every other move; and it can be
            uniquely optimal only if it is the mover's sole legal action.
  COROLLARY. Making Oops removals illegal cannot change any exact value, EXCEPT
            in a state where *every* legal action is an Oops. There the mover
            would instead have no legal action, and the shipped immediate-end
            rule scores the position by surviving-bead count -- which can differ
            from TRUE_LOSS. So the whole value question reduces to a single
            structural test: do such states exist?

What the experiment does, all exact:

  GATE 0.  Reproduce the published (4,4) census Oops anchors or abort.
  LEMMA.   Verify Lemmas 1-2 over every reachable state at (3,3)-(5,5), and
           classify every Oops as strictly-worse / tied-optimal / sole-legal.
  ABLATE.  Re-solve with ``forbid_oops()`` and compare value-by-value against
           the standard game over the whole ablated state space, plus root
           values on the reserve diagonal up to the full (14,14) game.
  MISERE.  Re-run the Finding 9 misere solves under the ablation. Relevance is
           objective-relative: a move that is the worst on the board for a
           player trying to win is attractive to a player trying to lose, so a
           rule can be inert under one objective and not another. Measured
           rather than argued.

The ablation chosen is "the self-destructive Removal is illegal". The other
candidate -- "the Removal is legal but the opponent's line does not count" --
is not well-posed: the line stays on the board, so the opponent simply wins at
the next terminal check, making it Oops-with-a-delay rather than no-Oops.

Run:  python -m experiments.oops_ablation           # (3,3)-(5,5) + (14,14)
      python -m experiments.oops_ablation 3 4       # pick the enumerated sizes
"""

import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

from collapse3.enumeration import reachable_states, solve_all
from collapse3.game import (
    TRUE_LOSS,
    GameState,
    apply_move,
    empty_state,
    evaluate_terminal,
    forbid_oops,
    get_legal_moves,
    is_oops,
    orient,
)
from collapse3.solver import NodeCapExceeded, capped_solve
from experiments._provenance import announce, write_result

NAME = "oops_ablation"

# Reproduce-or-abort: the published structural census at (4,4).
GATE0 = {"states_with_oops_removal": 16, "states_all_removals_oops": 0}

DIAGONAL_MAX = 14
ROOT_CAP = 30_000_000


def classify(r: int) -> Dict:
    """Exhaustive Oops census and Lemma 1-2 verification at reserves (r, r)."""
    root = empty_state(r, r)
    memo = solve_all(root)

    states_with_oops = 0
    states_all_removals_oops = 0
    states_all_moves_oops = 0          # the only case that can move a value
    oops_moves = 0
    oops_strictly_worse = 0
    oops_tied_optimal = 0
    oops_sole_legal_action = 0
    lemma1_violations: List[Tuple] = []
    lemma2_violations: List[Tuple] = []

    for s in memo:
        if evaluate_terminal(s) is not None:
            continue
        moves = get_legal_moves(s)
        if not moves:
            continue
        oops = [m for m in moves if is_oops(s, m)]
        if not oops:
            continue

        states_with_oops += 1
        oops_moves += len(oops)
        removals = [m for m in moves if m[0] == 'remove']
        if len(oops) == len(removals):
            states_all_removals_oops += 1
        if len(oops) == len(moves):
            states_all_moves_oops += 1

        vals = {m: orient(memo[apply_move(s, m)], s.turn) for m in moves}
        vstar = max(vals.values())

        for m in oops:
            # LEMMA 1: an Oops is worth exactly TRUE_LOSS to the mover.
            if vals[m] != TRUE_LOSS:
                lemma1_violations.append((s, m, vals[m]))
            if vals[m] < vstar:
                oops_strictly_worse += 1
            else:
                oops_tied_optimal += 1
                # LEMMA 2: tied-optimal means the position is already lost, so
                # EVERY move ties; an Oops is uniquely optimal only if alone.
                opt = [x for x, u in vals.items() if u == vstar]
                if vstar != TRUE_LOSS or len(opt) != len(moves):
                    lemma2_violations.append((s, m, vstar))
                if len(moves) == 1:
                    oops_sole_legal_action += 1

    return {
        "reserves": [r, r],
        "reachable_states": len(memo),
        "states_with_oops_removal": states_with_oops,
        "states_all_removals_oops": states_all_removals_oops,
        "states_all_legal_moves_oops": states_all_moves_oops,
        "oops_moves": oops_moves,
        "oops_strictly_worse": oops_strictly_worse,
        "oops_tied_optimal": oops_tied_optimal,
        "oops_uniquely_optimal": oops_sole_legal_action,
        "oops_ever_strictly_best": False if oops_moves else None,
        "lemma1_true_loss_violations": len(lemma1_violations),
        "lemma2_tied_implies_all_tied_violations": len(lemma2_violations),
    }


def ablate(r: int) -> Dict:
    """Re-solve with Oops removals illegal; compare every state to the standard
    game. The ablated move set is a subset of the standard one, so every
    ablated-reachable state is standard-reachable and directly comparable."""
    root = empty_state(r, r)
    std = solve_all(root)
    with forbid_oops():
        abl = solve_all(root)

    missing = [s for s in abl if s not in std]
    disagree = [(s, std[s], abl[s]) for s in abl if s in std and std[s] != abl[s]]

    return {
        "reserves": [r, r],
        "standard_states": len(std),
        "ablated_states": len(abl),
        "ablated_states_not_in_standard": len(missing),
        "value_disagreements": len(disagree),
        "root_standard": std[root],
        "root_ablated": abl[root],
        "root_unchanged": std[root] == abl[root],
    }


def root_diagonal(max_r: int) -> Dict[str, Dict[str, Optional[int]]]:
    """Root value at (r, r) with and without the Oops rule, up to the full game."""
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for r in range(1, max_r + 1):
        row: Dict[str, Optional[int]] = {}
        for label, ablated in (("standard", False), ("ablated", True)):
            try:
                if ablated:
                    with forbid_oops():
                        row[label] = capped_solve(empty_state(r, r), ROOT_CAP)[0]
                else:
                    row[label] = capped_solve(empty_state(r, r), ROOT_CAP)[0]
            except NodeCapExceeded:
                row[label] = None
        row["unchanged"] = (row["standard"] == row["ablated"])
        out[f"({r},{r})"] = row
        print(f"  ({r:>2},{r:>2}): standard {row['standard']}  "
              f"ablated {row['ablated']}  unchanged {row['unchanged']}")
    return out


def misere_ablation(sizes: Tuple[int, ...]) -> Dict:
    """Does the Oops rule matter under the MISERE objective? (Finding 9.)

    Under normal play an Oops is the worst move on the board. For a player trying
    to *lose* it is the best, and Finding 9 cites its mirror image -- reverse-Oops,
    a grandson removal whose cascade completes a line for grandpa -- as part of
    why a forced throw is impossible. ``forbid_oops()`` removes reverse-Oops too
    (from the mover's seat it is the same rule), so re-running the misere solves
    under the ablation tests that explanation directly instead of asserting it.
    """
    from experiments.sandbagging import (
        forced_throw_forceable,
        random_grandson_prob,
        weak_throw_forceable,
    )

    out: Dict[str, Dict] = {}
    for r in sizes:
        row: Dict[str, Dict] = {}
        for label in ("standard", "ablated"):
            ctx = forbid_oops() if label == "ablated" else nullcontext()
            with ctx:
                row[label] = {
                    "forced_throw_grandson_first": forced_throw_forceable(r, 0),
                    "forced_throw_grandson_second": forced_throw_forceable(r, 1),
                    "weak_throw_grandpa_first": weak_throw_forceable(r, 0),
                    "weak_throw_grandpa_second": weak_throw_forceable(r, 1),
                    "p_grandson_first_exact": str(random_grandson_prob(r, 0)),
                    "p_grandson_second_exact": str(random_grandson_prob(r, 1)),
                }
        keys = list(row["standard"])
        row["changed"] = {k: row["standard"][k] != row["ablated"][k] for k in keys}
        row["any_change"] = any(row["changed"].values())
        row["verdicts_changed"] = any(v for k, v in row["changed"].items()
                                      if k.startswith(("forced_", "weak_")))
        out[f"({r},{r})"] = row
        print(f"  ({r},{r}): forced throw {row['standard']['forced_throw_grandson_first']}"
              f"/{row['standard']['forced_throw_grandson_second']} -> "
              f"{row['ablated']['forced_throw_grandson_first']}"
              f"/{row['ablated']['forced_throw_grandson_second']}; "
              f"weak throw {row['standard']['weak_throw_grandpa_first']}"
              f"/{row['standard']['weak_throw_grandpa_second']} -> "
              f"{row['ablated']['weak_throw_grandpa_first']}"
              f"/{row['ablated']['weak_throw_grandpa_second']}; "
              f"any change: {row['any_change']} "
              f"(yes/no verdicts: {row['verdicts_changed']})")
        for seat in ("first", "second"):
            k = f"p_grandson_{seat}_exact"
            print(f"      P(grandson wins, {seat}): {row['standard'][k]} -> {row['ablated'][k]}"
                  f"  {'CHANGED' if row['changed'][k] else 'same'}")
    return out


def main(sizes: Tuple[int, ...] = (3, 4),
         census_only: Tuple[int, ...] = (5,),
         max_r: int = DIAGONAL_MAX) -> None:
    """``sizes`` get the full value-by-value ablation; ``census_only`` sizes get
    the census and lemma checks only, because holding two solved memos at once
    is not affordable at (5,5) (12.7M states). Their root values are still
    compared both ways by the diagonal sweep below."""
    t0 = time.time()

    print("Oops census + Lemma 1-2 verification (exhaustive)")
    census = {}
    for r in tuple(sizes) + tuple(census_only):
        c = classify(r)
        census[f"({r},{r})"] = c
        print(f"  ({r},{r}): {c['reachable_states']:,} states; "
              f"{c['states_with_oops_removal']} with an Oops "
              f"({c['oops_moves']} Oops moves) -- "
              f"{c['oops_strictly_worse']} strictly worse, "
              f"{c['oops_tied_optimal']} tied-optimal, "
              f"{c['oops_uniquely_optimal']} uniquely optimal; "
              f"all-legal-moves-Oops states: {c['states_all_legal_moves_oops']}")
        assert c["lemma1_true_loss_violations"] == 0, f"LEMMA 1 failed at ({r},{r})"
        assert c["lemma2_tied_implies_all_tied_violations"] == 0, \
            f"LEMMA 2 failed at ({r},{r})"

    # Gate 0: reproduce the published (4,4) anchors or abort.
    if 4 in tuple(sizes) + tuple(census_only):
        got = census["(4,4)"]
        for k, want in GATE0.items():
            assert got[k] == want, f"GATE 0 failed: {k} = {got[k]}, expected {want}"
        print(f"  gate 0 OK -- reproduced the published (4,4) anchors {GATE0}")

    print("\nAblation: Oops removals illegal, value-by-value (exhaustive)")
    ablation = {}
    for r in sizes:
        a = ablate(r)
        ablation[f"({r},{r})"] = a
        print(f"  ({r},{r}): {a['ablated_states']:,} ablated states; "
              f"{a['value_disagreements']} value disagreements; "
              f"root {a['root_standard']} -> {a['root_ablated']}")
        assert a["ablated_states_not_in_standard"] == 0

    print(f"\nRoot values with and without Oops, (1,1)..({max_r},{max_r}):")
    diagonal = root_diagonal(max_r)

    print("\nMisere ablation: does the rule matter to a player trying to LOSE?")
    misere = misere_ablation(tuple(sizes) + tuple(census_only))

    all_unchanged = (
        all(a["value_disagreements"] == 0 for a in ablation.values())
        and all(row["unchanged"] for row in diagonal.values())
    )
    never_strictly_best = all(c["oops_strictly_worse"] + c["oops_tied_optimal"]
                              == c["oops_moves"] for c in census.values())
    ever_forced = any(c["states_all_legal_moves_oops"] for c in census.values())

    payload = {
        "census": census,
        "ablation": ablation,
        "ablation_note": (
            "sizes in `census_only` are compared at the root by root_diagonal "
            "rather than value-by-value (two solved memos at (5,5) is 12.7M "
            "states twice)"),
        "root_diagonal": diagonal,
        "misere": misere,
        "verdict": {
            "oops_ever_strictly_best_move": not never_strictly_best,
            "oops_ever_forced_all_legal_moves": ever_forced,
            "exact_values_unchanged_by_ablation": all_unchanged,
            "full_game_root_unchanged": diagonal[f"({max_r},{max_r})"]["unchanged"],
            "misere_verdicts_unchanged": not any(
                m["verdicts_changed"] for m in misere.values()),
            "misere_probabilities_changed": any(
                m["any_change"] for m in misere.values()),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    v = payload["verdict"]
    print("\nVERDICT")
    print(f"  Oops ever the strictly best move : {v['oops_ever_strictly_best_move']}")
    print(f"  Oops ever forced (no alternative): {v['oops_ever_forced_all_legal_moves']}")
    print(f"  exact values unchanged by ablation: {v['exact_values_unchanged_by_ablation']}")
    print(f"  (14,14) root unchanged            : {v['full_game_root_unchanged']}")
    print(f"  misere yes/no verdicts unchanged  : {v['misere_verdicts_unchanged']}")
    print(f"  misere probabilities changed      : {v['misere_probabilities_changed']}")

    path = write_result(NAME, {"sizes": list(sizes), "diagonal_max": max_r,
                              "root_cap": ROOT_CAP}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args), ()) if args else main()
