"""Exact game value for symmetric reserves, with provenance.

Reproduces the ground-truth outcome of Collapse3 for reserves (r, r) up to a
configurable maximum, using the exact solver. This is the base fact every
downstream competence claim rests on, so it is recorded with full provenance.

Run:  python -m experiments.solve_reserves            # default up to r=5
      python -m experiments.solve_reserves 7          # up to r=7 (slower)
"""

import sys

from collapse3.solver import solve_reserves as solve_reserves_game
from experiments._provenance import announce, write_result

NAME = "solve_reserves"


def main(max_r: int = 5) -> None:
    config = {"reserves": [[r, r] for r in range(1, max_r + 1)]}
    rows = []
    print(f"Exact Collapse3 values for reserves (r, r), r=1..{max_r}")
    print("=" * 64)
    for r in range(1, max_r + 1):
        res = solve_reserves_game(r, r)
        rows.append({"reserves": [r, r], **res})
        print(f"  ({r},{r}): value={res['score_value']:>4}  "
              f"{res['optimal_outcome']:<32} nodes={res['nodes_visited']:>10}  "
              f"tt={res['table_size']:>9}  {res['time_seconds']}s")

    path = write_result(NAME, config, {"rows": rows})
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
