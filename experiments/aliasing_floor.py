"""Representation / aliasing regret floors at reserves (r, r).

Enumerates the whole state space and computes the irreducible regret floor of
the best memoryless policy under three lossy observations: hiding the cooldown,
hiding the reserves, and hiding both. The full-state floor is 0 by definition.

This is the ground-truth "cost of a missing state feature" -- the exact,
optimizer-independent lower bound that a learned agent on that observation can
never beat. (Reproduces the hide_cooldown ~= 0.0137 floor at (4,4).)

Run:  python -m experiments.aliasing_floor        # default r=4 (~1-2 min)
      python -m experiments.aliasing_floor 3        # smaller, fast
"""

import sys

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state
from experiments._provenance import announce, write_result

NAME = "aliasing_floor"


def main(r: int = 4) -> None:
    memo = solve_all(empty_state(r, r))

    rows = {}
    print(f"Aliasing regret floors, reserves ({r},{r}) -- {len(memo):,} states")
    print("=" * 72)
    print("  full-state floor: 0.0000 (by definition)")
    for name in ("hide_cooldown", "hide_reserves", "hide_both"):
        obs = OBSERVATIONS[name]
        res_all = regret_floor(memo, obs, p0_only=False)
        res_p0 = regret_floor(memo, obs, p0_only=True)
        rows[name] = {
            "floor_all_decisions": res_all.floor,
            "floor_p0_only": res_p0.floor,
            "obs_groups": res_all.obs_groups,
            "aliased_groups": res_all.aliased_groups,
            "conflict_groups": res_all.conflict_groups,
            "no_common_legal_action": res_all.no_common_legal_action,
            "max_states_per_obs": res_all.max_states_per_obs,
        }
        print(f"  {name:>14}: floor(all) {res_all.floor:.4f} | floor(P0) {res_p0.floor:.4f} | "
              f"aliased {res_all.aliased_groups:,}/{res_all.obs_groups:,} | "
              f"conflicts {res_all.conflict_groups:,}")

    path = write_result(NAME, {"reserves": [r, r]}, {"states": len(memo), "rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
