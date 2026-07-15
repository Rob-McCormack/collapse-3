"""The aliasing floor is a property of an *interface*, not a game or an agent.

For the reserve feature under ``hide_reserves``, the "cost of the missing
feature" is not one number -- it depends entirely on the interface the feature is
missing from. This experiment computes the exact ladder at reserves (r,r):

  rung                     interface the policy sees                          floor
  mask-blind               (board, turn, cooldown)                           exact
  mask-aware               + the legal-move list (what the trained QAgent     exact
                             actually sees -- QAgent.choose argmaxes over
                             get_legal_moves, so the mask is always in hand)
  + destroyed-bead memory  + reserve counts reconstructed from observed       exact (0)
                             removals

All three are exact, enumerated, uniform-over-decision-state floors (the Finding 4
machinery), so the ladder is **opponent-independent**. It is the honest synthesis
of Findings 4 and 7: the same missing feature costs 0.0805 to a (board, cooldown)
agent, ~0.0026 once the legal-move mask is in hand, and 0.0000 with memory.

The trained agents' *realized on-policy regret* is a fourth, different quantity --
trajectory- and opponent-dependent (Finding 7: ~0.0013 vs an eps-optimal
opponent but ~0.24 vs a random one). It is NOT a floor and is deliberately not
recomputed here; it is reported alongside the exact ladder in the docs, labelled
as the separate layer it is.

Run:  python -m experiments.interface_ladder 4        # default (4,4), ~1-2 min
      python -m experiments.interface_ladder 3 4      # add (3,3)
"""

import sys
import time

from collapse3.aliasing import OBSERVATIONS, memory_reserves_obs, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state, get_legal_moves
from experiments._provenance import announce, write_result

NAME = "interface_ladder"


def with_mask(base):
    """Fold the legal-move list into the observation (the mask-aware interface)."""
    return lambda s: (base(s), tuple(sorted(get_legal_moves(s))))


def ladder(r: int) -> dict:
    initial = (r, r)
    memo = solve_all(empty_state(r, r))
    blind = regret_floor(memo, OBSERVATIONS["hide_reserves"])
    aware = regret_floor(memo, with_mask(OBSERVATIONS["hide_reserves"]))
    memory = regret_floor(memo, memory_reserves_obs(initial))
    return {
        "states": len(memo),
        "mask_blind": blind.floor,
        "mask_aware": aware.floor,
        "mask_blind_plus_memory": memory.floor,
        "mask_aware_no_common_legal_action": aware.no_common_legal_action,
    }


def main(sizes=(4,)) -> None:
    by_size = {}
    print("Interface ladder for hide_reserves (exact, uniform-over-decision floors):")
    print(f"  {'res':>7} {'states':>10} {'mask-blind':>11} {'mask-aware':>11} {'+memory':>9}")
    for r in sizes:
        t0 = time.time()
        row = ladder(r)
        by_size[f"{r}_{r}"] = row
        assert row["mask_blind"] >= row["mask_aware"] >= row["mask_blind_plus_memory"], row
        assert row["mask_aware_no_common_legal_action"] == 0, "mask groups share a move-set; charity cannot fire"
        print(f"  ({r:>2},{r:<2}) {row['states']:>10,} {row['mask_blind']:>11.4f} "
              f"{row['mask_aware']:>11.4f} {row['mask_blind_plus_memory']:>9.4f}  [{time.time()-t0:.0f}s]")

    print("\nThe rungs are exact interface floors (opponent-independent). The trained")
    print("agents' realized on-policy regret is a separate, opponent-dependent layer")
    print("(Finding 7 / memory_floor): ~0.0013 vs eps-optimal, ~0.24 vs random at (4,4).")

    payload = {
        "feature": "hide_reserves",
        "sizes": list(sizes),
        "by_size": by_size,
        "on_policy_note": (
            "Realized on-policy regret is trajectory/opponent-dependent and lives in "
            "memory_floor.py / Finding 7; it is not an interface floor and is not "
            "recomputed here."
        ),
    }
    path = write_result(NAME, {"sizes": list(sizes), "feature": "hide_reserves"}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else (4,))
