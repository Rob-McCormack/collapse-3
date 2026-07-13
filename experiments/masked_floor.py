"""Aliasing floors when the legal-action mask is part of the observation.

External review pointed out that the published floors (`aliasing_floor.py`)
price a *mask-blind* interface: the policy sees only the lossy observation and
must pick one action legal in every aliased member. But the trained QAgent's
interface is richer -- it receives the state's legal-move list -- and that mask
leaks hidden state: under ``hide_cooldown`` the presence of removal moves often
reveals the cooldown bit; under ``hide_reserves`` the absence of placements
reveals an empty reserve.

This experiment recomputes the exact floors with the legal-action mask folded
into the observation key. Because every member of a masked group shares the
same legal-move set, the charity rule can never fire: masked floors are exact
optima and monotone under hiding more.

Run:  python -m experiments.masked_floor 3 4        # sizes (r,r)
"""

import sys
import time

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import empty_state, get_legal_moves
from experiments._provenance import announce, write_result

NAME = "masked_floor"


def with_mask(base):
    return lambda s: (base(s), tuple(sorted(get_legal_moves(s))))


def main(sizes=(3, 4)) -> None:
    by_size = {}
    for r in sizes:
        t0 = time.time()
        memo = solve_all(empty_state(r, r))
        print(f"\n({r},{r})  {len(memo):,} states  solve={time.time()-t0:.0f}s")
        rows = {}
        for name in ("hide_cooldown", "hide_reserves", "hide_both"):
            base = OBSERVATIONS[name]
            plain = regret_floor(memo, base)
            masked = regret_floor(memo, with_mask(base))
            rows[name] = {
                "floor_mask_blind": plain.floor,
                "floor_mask_aware": masked.floor,
                "masked_conflict_groups": masked.conflict_groups,
                "masked_no_common_legal_action": masked.no_common_legal_action,
            }
            print(f"  {name:>14}: mask-blind {plain.floor:.6f} -> mask-aware {masked.floor:.6f} "
                  f"(charity fired: {masked.no_common_legal_action})")
        by_size[f"{r}_{r}"] = {"states": len(memo), "rows": rows}

    path = write_result(NAME, {"sizes": list(sizes)}, {"by_size": by_size})
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else (3, 4))
