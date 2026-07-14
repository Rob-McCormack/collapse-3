"""The solvability horizon: exact grading of real games at full size (14,14).

The root of the 14-bead game is far beyond enumeration -- and stays OPEN here.
But real games are short: wins arrive long before the reserves empty. So a
node-capped exact solver (collapse3.solver.capped_solve) can walk a finished
game's trajectory from the END backwards and find the first ply at which the
remaining game fits inside the budget. Everything from that ply on is exactly
solved, and every move in that solved tail is graded against ground truth --
the same mover-perspective WDL regret used everywhere else in the repo.

Definitions:
  * horizon ply  -- the earliest ply (from the front) whose state solves
                    within the node cap using a fresh table. Plies before it
                    are unsolved; plies from it onward have exact values.
  * graded fraction -- (moves in the solved tail) / (moves in the game).
  * blunder      -- a solved-tail move with WDL regret > 0.

The consistency invariant (asserted over every solved tail, and in tests):
along a trajectory the exact value may change between consecutive solved
plies ONLY across a move graded with positive regret -- equivalently, the
mover-oriented value never improves across the mover's own move. A solver
whose transposition table stores cutoff values without bound flags violates
this (a +100 flipping to -100 across a zero-regret move); the invariant is
what caught that bug and it runs forever after.

Scope: horizons are trajectory- and agent-dependent. This experiment reports
distributions over a battery of pairings from the agent zoo, not one game.

Run:  python -m experiments.horizon [node_cap]
"""

import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from collapse3.agents import (
    Agent,
    HeuristicNPlyAgent,
    KimiAgentV1,
    KimiAgentV2,
    RandomAgent,
)
from collapse3.enumeration import wdl
from collapse3.game import (
    GameState,
    Move,
    apply_move,
    empty_state,
    evaluate_terminal,
    get_legal_moves,
    orient,
)
from collapse3.solver import NodeCapExceeded, capped_solve
from experiments._provenance import announce, write_result

NAME = "horizon"

RES = 14
DEFAULT_NODE_CAP = 4_000_000
SEEDS = 5


class EpsNoisedAgent(Agent):
    """Wrap a base agent: with probability ``epsilon`` play a uniform-random
    legal move, otherwise defer to the base policy."""

    name = "eps-noised"

    def __init__(self, base: Agent, epsilon: float, seed: int):
        import random
        self.base = base
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.name = f"{base.name}-eps{int(epsilon * 100)}"

    def choose(self, state: GameState) -> Move:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(get_legal_moves(state))
        return self.base.choose(state)


# name -> (factory(seed) -> (p0, p1), deterministic?)
PAIRINGS: Dict[str, Tuple[Callable[[int], Tuple[Agent, Agent]], bool]] = {
    "kimi-v2/kimi-v2": (lambda s: (KimiAgentV2(), KimiAgentV2()), True),
    "kimi-v1/kimi-v1": (lambda s: (KimiAgentV1(), KimiAgentV1()), True),
    "kimi-v1/kimi-v2": (lambda s: (KimiAgentV1(), KimiAgentV2()), True),
    "kimi-v2/kimi-v1": (lambda s: (KimiAgentV2(), KimiAgentV1()), True),
    "pos-2ply/pos-2ply": (lambda s: (HeuristicNPlyAgent(2, s), HeuristicNPlyAgent(2, s + 1)), False),
    "pos-1ply/pos-1ply": (lambda s: (HeuristicNPlyAgent(1, s), HeuristicNPlyAgent(1, s + 1)), False),
    "random/random": (lambda s: (RandomAgent(s), RandomAgent(s + 1)), False),
    "kimi-v2/random": (lambda s: (KimiAgentV2(), RandomAgent(s)), False),
    "random/kimi-v2": (lambda s: (RandomAgent(s), KimiAgentV2()), False),
    "kimi-v2-eps10/self": (
        lambda s: (EpsNoisedAgent(KimiAgentV2(), 0.10, s),
                   EpsNoisedAgent(KimiAgentV2(), 0.10, s + 1)),
        False,
    ),
}


def is_terminal(state: GameState) -> bool:
    return evaluate_terminal(state) is not None or not get_legal_moves(state)


def play_trajectory(p0: Agent, p1: Agent,
                    start: GameState) -> Tuple[List[GameState], List[Move]]:
    """Play one game; return (states, moves) with states[0] == start and
    states[-1] terminal (len(states) == len(moves) + 1)."""
    states, moves = [start], []
    state = start
    while not is_terminal(state):
        agent = p0 if state.turn == 0 else p1
        move = agent.choose(state)
        moves.append(move)
        state = apply_move(state, move)
        states.append(state)
    return states, moves


def analyze_game(states: List[GameState], moves: List[Move],
                 node_cap: int) -> dict:
    """Backward walk + exact tail grading + consistency invariant."""
    n = len(moves)
    values: Dict[int, int] = {}
    solve_stats: Dict[int, dict] = {}
    horizon = 0
    for i in range(n, -1, -1):
        t0 = time.time()
        try:
            v, nodes = capped_solve(states[i], node_cap)
        except NodeCapExceeded:
            horizon = i + 1
            solve_stats[i] = {"solved": False, "nodes": node_cap,
                              "seconds": round(time.time() - t0, 2)}
            break
        values[i] = v
        solve_stats[i] = {"solved": True, "value": v, "nodes": nodes,
                          "seconds": round(time.time() - t0, 2)}

    graded = []
    for i in range(horizon, n):
        mover = states[i].turn
        raw_drop = orient(values[i], mover) - orient(values[i + 1], mover)
        wdl_regret = wdl(values[i], mover) - wdl(values[i + 1], mover)
        # Consistency invariant. A violation means the capped solver served
        # a non-exact value (e.g. an unflagged alpha-beta cutoff).
        if raw_drop < 0:
            raise AssertionError(
                f"invariant violated at ply {i}: mover-oriented value improved "
                f"across the mover's own move ({values[i]} -> {values[i + 1]})")
        if (values[i] != values[i + 1]) != (raw_drop > 0):
            raise AssertionError(
                f"invariant violated at ply {i}: value changed "
                f"({values[i]} -> {values[i + 1]}) without positive regret")
        graded.append({"ply": i, "mover": mover, "move": list(moves[i]),
                       "value": values[i], "wdl_regret": wdl_regret,
                       "raw_regret": raw_drop})

    return {
        "plies": n,
        "final_value": values[n],
        "horizon_ply": horizon,
        "graded_moves": n - horizon,
        "graded_fraction": round((n - horizon) / n, 4) if n else 1.0,
        "tail_values": {str(i): values[i] for i in sorted(values)},
        "grading": graded,
        "blunders": sum(1 for g in graded if g["wdl_regret"] > 0),
        "solve_stats": {str(i): solve_stats[i] for i in sorted(solve_stats)},
    }


def run_pairing(pairing: str, node_cap: int, seeds: int) -> List[dict]:
    factory, deterministic = PAIRINGS[pairing]
    start = empty_state(RES, RES)
    games = []
    for s in range(1 if deterministic else seeds):
        seed = 1000 + 17 * s
        p0, p1 = factory(seed)
        states, moves = play_trajectory(p0, p1, start)
        rec = analyze_game(states, moves, node_cap)
        rec["seed"] = None if deterministic else seed
        games.append(rec)
        print(f"  {pairing} seed={rec['seed']}: {rec['plies']} plies, "
              f"final {rec['final_value']:+d}, horizon ply {rec['horizon_ply']}, "
              f"graded {rec['graded_fraction']:.0%}, "
              f"blunders {rec['blunders']}")
    return games


def main(node_cap: int = DEFAULT_NODE_CAP) -> None:
    results = {}
    t0 = time.time()
    for pairing in PAIRINGS:
        print(f"{pairing}:")
        games = run_pairing(pairing, node_cap, SEEDS)
        results[pairing] = {
            "games": games,
            "lengths": [g["plies"] for g in games],
            "horizons": [g["horizon_ply"] for g in games],
            "graded_fractions": [g["graded_fraction"] for g in games],
            "blunder_counts": [g["blunders"] for g in games],
        }

    print(f"\n{'pairing':<22}{'len':>12}{'horizon':>12}{'graded':>10}{'blunders':>10}")
    for pairing, r in results.items():
        def fmt(xs):
            return "/".join(str(x) for x in xs)
        print(f"{pairing:<22}{fmt(r['lengths']):>12}{fmt(r['horizons']):>12}"
              f"{fmt([f'{f:.0%}' for f in r['graded_fractions']]):>10}"
              f"{fmt(r['blunder_counts']):>10}")

    path = write_result(NAME, {
        "reserves": [RES, RES],
        "node_cap": node_cap,
        "seeds_per_stochastic_pairing": SEEDS,
        "pairings": sorted(PAIRINGS),
    }, {"pairings": results,
        "elapsed_seconds": round(time.time() - t0, 1)})
    announce(NAME, path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NODE_CAP)
