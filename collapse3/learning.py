"""Tabular reinforcement learning with pluggable (possibly lossy) observations.

A memoryless tabular Q-learner that trains from **terminal reward only** and
has **zero oracle access** during training. Its observation of a state can be
made deliberately lossy -- hiding the cooldown flags and/or the reserves -- so
we can measure how much competence a learned agent loses purely to an
inadequate state representation (the representation/aliasing study).

Agents can also carry a tiny **memory** of observed removals (destroyed-bead
counts), maintained purely from moves the agent sees -- never read from hidden
state fields.

All play uses the shipped immediate-game-end rule (no pass).
"""

import random
from typing import Callable, Dict, Optional, Tuple, Union

from .game import (
    Board,
    GameState,
    Move,
    apply_move,
    attrition_value,
    evaluate_terminal,
    get_legal_moves,
    orient,
)
from .agents import Agent

# Observation functions: full state vs. history-created fields hidden.
OBSERVATIONS: Dict[str, Callable[[GameState], Tuple]] = {
    "full": lambda s: (s.board, s.res, s.turn, s.cooldown),
    "hide_cooldown": lambda s: (s.board, s.res, s.turn),
    "hide_reserves": lambda s: (s.board, s.turn, s.cooldown),
    "hide_both": lambda s: (s.board, s.turn),
}

DestroyedCounts = Tuple[int, int]  # (my destroyed, opponent destroyed) from agent seat
MemoryObsFn = Callable[[GameState, DestroyedCounts], Tuple]
ObsMode = Union[str, MemoryObsFn]

OpponentPolicy = Callable[[GameState], Move]
QTable = Dict[Tuple, float]


def beads_on_board(board: Board, player: int) -> int:
    return sum(col.count(player) for col in board)


def destroyed_from_state(state: GameState, initial_res: Tuple[int, int]) -> DestroyedCounts:
    """Beads removed from the board per player, derivable from a full snapshot."""
    on0 = beads_on_board(state.board, 0)
    on1 = beads_on_board(state.board, 1)
    return (initial_res[0] - state.res[0] - on0, initial_res[1] - state.res[1] - on1)


def memory_reserves_obs(initial_res: Tuple[int, int]) -> MemoryObsFn:
    """Reserve-blind observation plus destroyed counts (information-equivalent to full)."""

    def obs(state: GameState, mem: DestroyedCounts) -> Tuple:
        return (state.board, state.turn, state.cooldown, mem)

    return obs


MEMORY_OBSERVATIONS: Dict[str, Callable[[Tuple[int, int]], MemoryObsFn]] = {
    "hide_reserves": lambda initial: (lambda s, _mem: (s.board, s.turn, s.cooldown)),
    "hide_reserves_memory": memory_reserves_obs,
}


def _wdl(value: int, player: int) -> int:
    u = orient(value, player)
    return 1 if u > 0 else (-1 if u < 0 else 0)


def optimal_opponent(memo: Dict[GameState, int], epsilon: float, rng: random.Random) -> OpponentPolicy:
    """An (epsilon-noisy) exact opponent that plays from a precomputed value map.

    A little noise is important: against a purely deterministic optimal opponent
    the game collapses to one line and aliased states are never visited, so the
    representation floor would read ~0 for every observation.
    """
    def policy(state: GameState) -> Move:
        moves = get_legal_moves(state)
        if epsilon and rng.random() < epsilon:
            return rng.choice(moves)
        mover = state.turn
        best, best_u = None, None
        for m in moves:
            u = orient(memo[apply_move(state, m)], mover)
            if best_u is None or u > best_u:
                best, best_u = m, u
        return best
    return policy


def advance_to_agent(state: GameState, agent_seat: int, opponent: OpponentPolicy):
    """Play the opponent until it is the agent's turn or the game ends.

    Returns ``(state, reward_or_None)`` with reward in win/draw/loss units from
    the agent's seat (``None`` while the agent still has a decision to make).
    """
    while True:
        t = evaluate_terminal(state)
        if t is not None:
            return state, float(_wdl(t, agent_seat))
        moves = get_legal_moves(state)
        if not moves:                       # immediate end -> attrition
            return state, float(_wdl(attrition_value(state.board), agent_seat))
        if state.turn == agent_seat:
            return state, None
        state = apply_move(state, opponent(state))


def apply_move_tracked(
    state: GameState,
    move: Move,
    mem: DestroyedCounts,
    agent_seat: int,
) -> DestroyedCounts:
    """Update destroyed-bead memory when a removal is observed."""
    if move[0] != "remove":
        return mem
    my_d, opp_d = mem
    if state.turn == agent_seat:
        return (my_d, opp_d + 1)
    return (my_d + 1, opp_d)


def advance_to_agent_with_memory(
    state: GameState,
    mem: DestroyedCounts,
    agent_seat: int,
    opponent: OpponentPolicy,
):
    """Like :func:`advance_to_agent`, but track observed removals in ``mem``."""
    while True:
        t = evaluate_terminal(state)
        if t is not None:
            return state, mem, float(_wdl(t, agent_seat))
        moves = get_legal_moves(state)
        if not moves:
            return state, mem, float(_wdl(attrition_value(state.board), agent_seat))
        if state.turn == agent_seat:
            return state, mem, None
        move = opponent(state)
        mem = apply_move_tracked(state, move, mem, agent_seat)
        state = apply_move(state, move)


def train_q_with_memory(
    root: GameState,
    obs_fn: MemoryObsFn,
    episodes: int,
    opponent: OpponentPolicy,
    agent_seat: int = 0,
    seed: int = 0,
    eps0: float = 0.4,
    lr0: float = 0.25,
) -> QTable:
    """Tabular Q-learning with a (state, memory) observation key."""
    rng = random.Random(seed)
    Q: QTable = {}

    for ep in range(episodes):
        frac = ep / episodes if episodes else 1.0
        eps = max(0.05, eps0 * (1 - frac))
        lr = max(0.02, lr0 * (1 - frac))

        state, mem, reward = advance_to_agent_with_memory(
            root, (0, 0), agent_seat, opponent)
        prev: Optional[Tuple] = None
        while reward is None:
            key = obs_fn(state, mem)
            moves = get_legal_moves(state)
            if rng.random() < eps:
                a = rng.choice(moves)
            else:
                a = max(moves, key=lambda m: (Q.get((key, m), 0.0), -moves.index(m)))
            if prev is not None:
                nmax = max(Q.get((key, m), 0.0) for m in moves)
                Q[prev] = Q.get(prev, 0.0) + lr * (nmax - Q.get(prev, 0.0))
            prev = (key, a)
            mem = apply_move_tracked(state, a, mem, agent_seat)
            state, mem, reward = advance_to_agent_with_memory(
                apply_move(state, a), mem, agent_seat, opponent)
        if prev is not None:
            Q[prev] = Q.get(prev, 0.0) + lr * (reward - Q.get(prev, 0.0))

    return Q


class MemoryQAgent(Agent):
    """Greedy policy over a trained Q-table keyed by (observation, memory)."""

    def __init__(
        self,
        Q: QTable,
        obs_fn: MemoryObsFn,
        mem: DestroyedCounts = (0, 0),
        name: Optional[str] = None,
    ):
        self.Q = Q
        self.obs_fn = obs_fn
        self.mem = mem
        self.name = name or "learned(memory)"

    def choose(self, state: GameState) -> Move:
        key = self.obs_fn(state, self.mem)
        moves = get_legal_moves(state)
        return max(moves, key=lambda m: (self.Q.get((key, m), 0.0), -moves.index(m)))

    def with_memory(self, mem: DestroyedCounts) -> "MemoryQAgent":
        return MemoryQAgent(self.Q, self.obs_fn, mem, self.name)


def train_q(
    root: GameState,
    obs_mode: str,
    episodes: int,
    opponent: OpponentPolicy,
    agent_seat: int = 0,
    seed: int = 0,
    eps0: float = 0.4,
    lr0: float = 0.25,
) -> QTable:
    """Tabular Q-learning, terminal reward only, gamma = 1.

    Decaying epsilon/learning-rate schedules. ``opponent`` moves for the seat the
    agent does not occupy. Returns a Q-table keyed by ``(obs_key, move)``.
    """
    rng = random.Random(seed)
    obs = OBSERVATIONS[obs_mode]
    Q: QTable = {}

    for ep in range(episodes):
        frac = ep / episodes if episodes else 1.0
        eps = max(0.05, eps0 * (1 - frac))
        lr = max(0.02, lr0 * (1 - frac))

        state, reward = advance_to_agent(root, agent_seat, opponent)
        prev: Optional[Tuple] = None
        while reward is None:
            key = obs(state)
            moves = get_legal_moves(state)
            if rng.random() < eps:
                a = rng.choice(moves)
            else:
                a = max(moves, key=lambda m: (Q.get((key, m), 0.0), -moves.index(m)))
            if prev is not None:
                nmax = max(Q.get((key, m), 0.0) for m in moves)
                Q[prev] = Q.get(prev, 0.0) + lr * (nmax - Q.get(prev, 0.0))
            prev = (key, a)
            state, reward = advance_to_agent(apply_move(state, a), agent_seat, opponent)
        if prev is not None:
            Q[prev] = Q.get(prev, 0.0) + lr * (reward - Q.get(prev, 0.0))

    return Q


class QAgent(Agent):
    """Greedy policy over a trained Q-table, using the same observation."""

    def __init__(self, Q: QTable, obs_mode: str = "full", name: Optional[str] = None):
        self.Q = Q
        self.obs = OBSERVATIONS[obs_mode]
        self.name = name or f"learned({obs_mode})"

    def choose(self, state: GameState) -> Move:
        key = self.obs(state)
        moves = get_legal_moves(state)
        return max(moves, key=lambda m: (self.Q.get((key, m), 0.0), -moves.index(m)))
