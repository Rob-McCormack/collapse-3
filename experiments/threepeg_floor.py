"""Three-Peg Collapse: the complete aliasing-floor curve across every reserve size.

Three-Peg Collapse is a **sibling** of Collapse3, not a miniature: placements are
restricted to the single row of pegs (0, 1, 2) and *every other mechanic is
untouched* -- stacking, gravity, the Collapse/removal with all five legality
conditions, cooldown, the Oops rule, line wins, attrition, and the repo rule
(no legal action -> immediate attrition end). Win detection needs no change:
pegs 3-8 stay empty, so only 8 of the full board's 49 winning lines can ever
fire (3 verticals, the row at 3 levels, 2 staircases). Because play lives on one
row, the game is fully enumerable at **every** reserve size, including the real
(14,14) (94,824 reachable states, ~2s) -- which the full board never is past
(5,5).

That turns Finding 4's two enumerable points (hide_reserves 0.0805 at (4,4),
0.1677 at (5,5)) into a complete 13-point curve for the sibling, and lets us see
its *shape*: hide_reserves grows steeply then **saturates** near ~0.25, and
hide_cooldown is **non-monotonic** (peaks at (7,7), then declines). These are
PARALLEL evidence about the same mechanics under different geometry; they are
NOT facts about Collapse3 and must never be quoted as such.

Run:  python -m experiments.threepeg_floor            # full sweep 2..14 (~30-60s)
      python -m experiments.threepeg_floor 2 3 4 5    # a subset of sizes
"""

import sys
import time

from collapse3.aliasing import OBSERVATIONS, regret_floor
from collapse3.enumeration import solve_all
from collapse3.game import WINNING_LINES, apply_move, empty_state, placement_pegs
from experiments._provenance import announce, write_result

NAME = "threepeg_floor"
PLACEMENT_ROW = (0, 1, 2)

_LABEL = {100: "P0 line", 10: "P0 attrition", 0: "draw", -10: "P1 attrition", -100: "P1 line"}

# --- A2 reproduce gate (deterministic; ABORT on any mismatch = engine drift) ---
# res -> (reachable states, decision states, hide_cooldown floor, hide_reserves floor)
EXPECTED_FLOORS = {
    2: (154, 55, 0.0000, 0.0000),
    3: (1108, 616, 0.0049, 0.0097),
    4: (4352, 2869, 0.0199, 0.0641),
    5: (10505, 7435, 0.0371, 0.1235),
    6: (18624, 13614, 0.0422, 0.1643),
    7: (27773, 20674, 0.0424, 0.1905),
    8: (37338, 28080, 0.0401, 0.2090),
    9: (46919, 35562, 0.0390, 0.2209),
    10: (56500, 43044, 0.0383, 0.2286),
    11: (66081, 50526, 0.0379, 0.2341),
    12: (75662, 58008, 0.0375, 0.2382),
    13: (85243, 65490, 0.0373, 0.2413),
    14: (94824, 72972, 0.0371, 0.2438),
}
# res -> (root value, [value after place on peg0, peg1, peg2])
EXPECTED_OPENINGS = {
    1: (0, [0, 0, 0]),
    2: (0, [0, 0, 0]),
    3: (10, [10, 10, 10]),
    4: (10, [10, 10, 10]),
    5: (10, [0, 10, 0]),        # centre uniquely wins
    6: (10, [10, 0, 10]),       # ends win, centre draws
    7: (10, [10, -100, 10]),    # ends win, centre LOSES to a P1 line
    8: (10, [10, -100, 10]),
    9: (10, [10, -100, 10]),
    10: (10, [10, -100, 10]),
    11: (10, [10, -100, 10]),
    12: (10, [10, -100, 10]),
    13: (10, [10, -100, 10]),
    14: (10, [10, -100, 10]),
}
_FLOOR_TOL = 5e-5


def winning_line_census(allowed=PLACEMENT_ROW):
    """How many of the full board's 49 winning-line configurations can fire when
    beads live only on ``allowed`` pegs. Verticals (9) + flats-per-level (8x3=24)
    + staircases (8x2=16) = 49; a line is live iff all its pegs are ``allowed``.
    """
    allowed = set(allowed)
    verticals = list(range(9))
    flats = [(line, z) for line in WINNING_LINES for z in range(3)]
    stairs = [(line, d) for line in WINNING_LINES for d in ("asc", "desc")]
    live_vert = [p for p in verticals if p in allowed]
    live_flat = [(l, z) for (l, z) in flats if set(l) <= allowed]
    live_stair = [(l, d) for (l, d) in stairs if set(l) <= allowed]
    return {
        "total": len(verticals) + len(flats) + len(stairs),
        "live": len(live_vert) + len(live_flat) + len(live_stair),
        "verticals": len(live_vert),
        "row_levels": len(live_flat),
        "staircases": len(live_stair),
    }


def opening_values(r):
    """Root value and the value after each first placement, P0-perspective."""
    memo = solve_all(empty_state(r, r))
    root = empty_state(r, r)
    return memo[root], [memo[apply_move(root, ("place", peg))] for peg in PLACEMENT_ROW]


def analyse_size(r):
    """Enumerate (r,r) under the three-peg restriction and compute every floor."""
    memo = solve_all(empty_state(r, r))
    rows = {}
    decisions = 0
    for name in ("hide_cooldown", "hide_reserves", "hide_both"):
        res = regret_floor(memo, OBSERVATIONS[name])
        decisions = res.n_decisions  # census size is identical across observations
        rows[name] = {
            "floor": res.floor,
            "obs_groups": res.obs_groups,
            "aliased_groups": res.aliased_groups,
            "conflict_groups": res.conflict_groups,
            "no_common_legal_action": res.no_common_legal_action,
        }
    return {"states": len(memo), "decisions": decisions, "rows": rows}


def main(sizes=tuple(range(2, 15))):
    census = winning_line_census()
    assert census == {"total": 49, "live": 8, "verticals": 3, "row_levels": 3, "staircases": 2}, census

    with placement_pegs(PLACEMENT_ROW):
        print(f"Three-Peg Collapse -- sibling variant (placements on pegs {list(PLACEMENT_ROW)})")
        print(f"Live winning lines: {census['live']} of {census['total']} "
              f"({census['verticals']} verticals, {census['row_levels']} row-levels, "
              f"{census['staircases']} staircases)\n")

        # Opening gate (sizes 1..14, independent of the floor sweep).
        openings = {}
        print("Opening gate (root value / value after place on peg 0,1,2):")
        for r in range(1, 15):
            root_v, place_v = opening_values(r)
            openings[f"{r}_{r}"] = {
                "root": root_v, "root_label": _LABEL[root_v],
                "placements": [{"peg": p, "value": v, "label": _LABEL[v]}
                               for p, v in zip(PLACEMENT_ROW, place_v)],
            }
            exp = EXPECTED_OPENINGS.get(r)
            if exp is not None:
                assert root_v == exp[0] and place_v == exp[1], (
                    f"OPENING GATE MISMATCH at ({r},{r}): got root={root_v} places={place_v}, "
                    f"expected root={exp[0]} places={exp[1]} -- ENGINE DRIFT, aborting.")
            tag = "" if exp is None else " OK"
            print(f"  ({r},{r}): root {_LABEL[root_v]:>12} | "
                  f"peg0 {_LABEL[place_v[0]]:>12}  peg1 {_LABEL[place_v[1]]:>12}  "
                  f"peg2 {_LABEL[place_v[2]]:>12}{tag}")

        # Floor sweep (sizes 2..14).
        print("\nAliasing-floor curve (uniform over decision states, WDL units):")
        print(f"  {'res':>7} {'states':>8} {'decisions':>10} {'hide_cd':>9} "
              f"{'d(cd)':>8} {'hide_res':>9} {'d(res)':>8} {'hide_both':>10} {'charity(cd,res)':>16}")
        by_size = {}
        prev_cd = prev_res = None
        exactness_holds = True
        for r in sizes:
            t0 = time.time()
            info = analyse_size(r)
            rows = info["rows"]
            cd = rows["hide_cooldown"]["floor"]
            rv = rows["hide_reserves"]["floor"]
            hb = rows["hide_both"]["floor"]
            cd_no = rows["hide_cooldown"]["no_common_legal_action"]
            rv_no = rows["hide_reserves"]["no_common_legal_action"]
            if cd_no or rv_no:
                exactness_holds = False
            d_cd = None if prev_cd is None else round(cd - prev_cd, 4)
            d_rv = None if prev_res is None else round(rv - prev_res, 4)

            exp = EXPECTED_FLOORS.get(r)
            if exp is not None:
                assert info["states"] == exp[0], f"state count drift at ({r},{r}): {info['states']} != {exp[0]}"
                assert info["decisions"] == exp[1], f"decision count drift at ({r},{r}): {info['decisions']} != {exp[1]}"
                assert abs(cd - exp[2]) < _FLOOR_TOL, f"hide_cooldown drift at ({r},{r}): {cd} != {exp[2]}"
                assert abs(rv - exp[3]) < _FLOOR_TOL, f"hide_reserves drift at ({r},{r}): {rv} != {exp[3]}"

            by_size[f"{r}_{r}"] = {
                "states": info["states"],
                "decisions": info["decisions"],
                "rows": rows,
                "hide_cooldown_increment": d_cd,
                "hide_reserves_increment": d_rv,
            }
            print(f"  ({r:>2},{r:<2}) {info['states']:>8,} {info['decisions']:>10,} "
                  f"{cd:>9.4f} {('' if d_cd is None else f'{d_cd:+.4f}'):>8} "
                  f"{rv:>9.4f} {('' if d_rv is None else f'{d_rv:+.4f}'):>8} "
                  f"{hb:>10.4f} {f'({cd_no},{rv_no})':>16}  [{time.time()-t0:.0f}s]")
            prev_cd, prev_res = cd, rv

        cd_curve = [(r, by_size[f"{r}_{r}"]["rows"]["hide_cooldown"]["floor"]) for r in sizes]
        peak = max(cd_curve, key=lambda x: x[1])
        last_hr = by_size[f"{sizes[-1]}_{sizes[-1]}"]["rows"]["hide_reserves"]["floor"]
        print(f"\nExactness lemma (single-hide charity never fires) holds at every size: {exactness_holds}")
        print(f"hide_cooldown peak: {peak[1]:.4f} at ({peak[0]},{peak[0]}) "
              f"-> then declines (non-monotonic).")
        print(f"hide_reserves: rises to {last_hr:.4f} at "
              f"({sizes[-1]},{sizes[-1]}) with increments collapsing toward the asymptote.")

    payload = {
        "variant": "Three-Peg Collapse",
        "placement_pegs": list(PLACEMENT_ROW),
        "winning_line_census": census,
        "sizes": list(sizes),
        "openings": openings,
        "by_size": by_size,
        "exactness_lemma_holds": exactness_holds,
        "hide_cooldown_peak": {"size": peak[0], "floor": peak[1]},
    }
    path = write_result(NAME, {"sizes": list(sizes), "placement_pegs": list(PLACEMENT_ROW)}, payload)
    announce(NAME, path)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(tuple(args) if args else tuple(range(2, 15)))
