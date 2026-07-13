"""Reference policies for Collapse3.

Every agent implements ``choose(state) -> Move`` and carries a ``name``. All
stochastic choices go through an explicit, seeded ``random.Random`` so runs are
reproducible: the same (agent, seed, opponent) always produces the same game.

Agents provided:
    OptimalAgent -- plays a game-theoretically optimal move (via the oracle).
    RandomAgent  -- uniform over legal moves.
    NPlyAgent    -- depth-limited alpha-beta with a weak attrition heuristic;
                    ``depth=1`` is the classic "one-ply" greedy searcher.
    MyopicAgent  -- depth-1 greedy on a naive own-alignment heuristic that
                    ignores opponent threats (models "off-distribution weirdness").
"""

import random
from typing import List, Optional

from .game import (
    GameState,
    Move,
    WINNING_LINES,
    apply_move,
    attrition_value,
    evaluate_terminal,
    get_legal_moves,
    orient,
)
from .oracle import Oracle


class Agent:
    name: str = "agent"

    def choose(self, state: GameState) -> Move:  # pragma: no cover - interface
        raise NotImplementedError


def _seeded(seed: Optional[int]) -> random.Random:
    return random.Random(seed)


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.name = f"random(seed={seed})"
        self.rng = _seeded(seed)

    def choose(self, state: GameState) -> Move:
        return self.rng.choice(get_legal_moves(state))


class OptimalAgent(Agent):
    """Plays an optimal move; ties broken by a seeded RNG for variety.

    With ``epsilon > 0`` it plays a uniformly random legal move that fraction of
    the time (an "eps-optimal" opponent). A little noise matters: a purely
    deterministic optimal opponent funnels every game down one line, so
    off-distribution / aliased states are never visited.
    """

    def __init__(self, seed: Optional[int] = None, epsilon: float = 0.0,
                 oracle: Optional[Oracle] = None):
        self.name = f"optimal(seed={seed}, eps={epsilon})"
        self.epsilon = epsilon
        self.rng = _seeded(seed)
        self.oracle = oracle or Oracle()

    def choose(self, state: GameState) -> Move:
        moves = get_legal_moves(state)
        if not moves:
            raise ValueError("choose() called on a state with no legal moves")
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(moves)
        label = self.oracle.label_state(state)
        best = [m.move for m in label.moves if m.is_optimal]
        return self.rng.choice(best)


# The 49 winning lines as (peg, level) cell triples: 9 verticals, 24 flats
# (rows/cols/diagonals on each level), 16 staircases.
ALL_LINES = []
for _peg in range(9):
    ALL_LINES.append(((_peg, 0), (_peg, 1), (_peg, 2)))
for _z in range(3):
    for (_a, _b, _c) in WINNING_LINES:
        ALL_LINES.append(((_a, _z), (_b, _z), (_c, _z)))
for (_a, _b, _c) in WINNING_LINES:
    ALL_LINES.append(((_a, 0), (_b, 1), (_c, 2)))
    ALL_LINES.append(((_a, 2), (_b, 1), (_c, 0)))

# Count-indexed line weights; index 3 (a completed line) dominates but is only
# reached on terminal boards, which the agent handles separately.
_LINE_WEIGHT = (0, 1, 10, 1000)


def board_heuristic(board, me: int) -> int:
    """Static score of a board from ``me``'s perspective: live-line potential
    (weighted by how many of my beads share an unblocked line) plus material."""
    opp = 1 - me
    score = 0
    for line in ALL_LINES:
        mine = theirs = 0
        for (peg, z) in line:
            if len(board[peg]) > z:
                if board[peg][z] == me:
                    mine += 1
                else:
                    theirs += 1
        if mine and theirs:
            continue  # dead line: neither can complete it
        score += _LINE_WEIGHT[mine] - _LINE_WEIGHT[theirs]
    beads_me = sum(p.count(me) for p in board)
    beads_opp = sum(p.count(opp) for p in board)
    return score + 2 * (beads_me - beads_opp)


class HeuristicOnePlyAgent(Agent):
    """Looks exactly one move ahead: take an immediate win, avoid an immediate
    loss, else pick the move maximising a static board heuristic. No search
    depth, no learning, no memory."""

    def __init__(self, seed: Optional[int] = None):
        self.name = f"one-ply-heuristic(seed={seed})"
        self.rng = _seeded(seed)

    def choose(self, state: GameState) -> Move:
        p = state.turn
        wins, safe, losing = [], [], []
        for m in get_legal_moves(state):
            child = apply_move(state, m)
            t = evaluate_terminal(child)
            if t is not None:
                u = orient(t, p)
                if u > 0:
                    wins.append(m)
                elif u < 0:
                    losing.append((m, child))
                else:
                    safe.append((m, child))
            else:
                safe.append((m, child))
        if wins:
            return self.rng.choice(wins)
        pool = safe if safe else losing
        best_score = max(board_heuristic(c.board, p) for _, c in pool)
        best_moves = [m for m, c in pool if board_heuristic(c.board, p) == best_score]
        return self.rng.choice(best_moves)


def _attrition_heuristic(state: GameState) -> float:
    """Weak leaf evaluation (P0 perspective): surviving-bead difference."""
    p0 = sum(peg.count(0) for peg in state.board)
    p1 = sum(peg.count(1) for peg in state.board)
    return float(p0 - p1)


class NPlyAgent(Agent):
    """Depth-limited alpha-beta searcher with a weak heuristic at the horizon.

    Not game-theoretically optimal: it cannot see terminal facts beyond ``depth``
    plies and falls back to a bead-count heuristic, so it makes exactly the kind
    of horizon errors we want to measure.
    """

    def __init__(self, depth: int = 1, seed: Optional[int] = None):
        assert depth >= 1
        self.depth = depth
        self.name = f"{depth}-ply(seed={seed})"
        self.rng = _seeded(seed)

    def _search(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        terminal = evaluate_terminal(state)
        if terminal is not None:
            return float(terminal)
        if depth == 0:
            return _attrition_heuristic(state)
        moves = get_legal_moves(state)
        if not moves:
            return _attrition_heuristic(state)
        if state.turn == 0:
            best = -1e9
            for move in moves:
                best = max(best, self._search(apply_move(state, move), depth - 1, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        else:
            best = 1e9
            for move in moves:
                best = min(best, self._search(apply_move(state, move), depth - 1, alpha, beta))
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

    def choose(self, state: GameState) -> Move:
        moves = get_legal_moves(state)
        mover = state.turn
        scored = []
        for move in moves:
            v = self._search(apply_move(state, move), self.depth - 1, -1e9, 1e9)
            scored.append((orient(int(round(v)) if abs(v) >= 1e8 else v, mover), move))
        best_score = max(s for s, _ in scored)
        best_moves = [m for s, m in scored if s == best_score]
        return self.rng.choice(best_moves)


class MyopicAgent(Agent):
    """Depth-1 greedy on a naive own-alignment score, ignoring opponent threats.

    For each legal move it scores the resulting board by how much of its own
    material sits on shared winning lines, taking an immediate win if offered.
    It never reasons about the opponent's reply, so it happily walks into
    off-distribution states -- useful as an adversarially "weird" opponent.
    """

    def __init__(self, seed: Optional[int] = None):
        self.name = f"myopic(seed={seed})"
        self.rng = _seeded(seed)

    @staticmethod
    def _alignment_score(state: GameState, player: int) -> float:
        board = state.board
        score = 0.0
        # Flat lines per level.
        for z in range(3):
            for line in WINNING_LINES:
                own = 0
                blocked = False
                for peg in line:
                    if len(board[peg]) > z:
                        if board[peg][z] == player:
                            own += 1
                        else:
                            blocked = True
                if not blocked:
                    score += own * own
        # Verticals.
        for peg in board:
            own = sum(1 for b in peg if b == player)
            opp = len(peg) - own
            if opp == 0:
                score += own * own
        return score

    def choose(self, state: GameState) -> Move:
        moves = get_legal_moves(state)
        mover = state.turn
        best_moves: List[Move] = []
        best_score = -1e18
        for move in moves:
            child = apply_move(state, move)
            t = evaluate_terminal(child)
            if t is not None and orient(t, mover) >= 100:
                return move  # take an immediate true win
            s = self._alignment_score(child, mover)
            if s > best_score:
                best_score = s
                best_moves = [move]
            elif s == best_score:
                best_moves.append(move)
        return self.rng.choice(best_moves)


# ---------------------------------------------------------------------------
# KimiAgent: an externally proposed, human-readable "basic strategy", frozen.
# ---------------------------------------------------------------------------

# Peg priority for placements: centre, corners, edges.
KIMI_PRIORITY = (4, 0, 2, 6, 8, 1, 3, 5, 7)


def _end_value(state: GameState) -> Optional[int]:
    """Terminal value of ``state`` under the repo rule, else None.

    Includes the no-legal-move -> immediate attrition end, so "win-in-1"
    means the move genuinely ends the game in the mover's favour.
    """
    t = evaluate_terminal(state)
    if t is not None:
        return t
    if not get_legal_moves(state):
        return attrition_value(state.board)
    return None


class _KimiBase(Agent):
    """Shared machinery for the two frozen formalizations (see subclasses)."""

    def _win_in_1(self, state: GameState) -> Optional[Move]:
        # Rule 1: first legal move (engine order) whose child is a terminal
        # win for the mover. This is the ONLY rule that prefers removals.
        p = state.turn
        for m in get_legal_moves(state):
            t = _end_value(apply_move(state, m))
            if t is not None and orient(t, p) > 0:
                return m
        return None

    def _tiers(self, state: GameState) -> List[List[Move]]:
        """T0 empty pegs / T1 opponent-topped / T2 self-topped / T3 removals.

        Placements within a tier follow KIMI_PRIORITY; removals engine order.
        """
        p = state.turn
        board = state.board
        legal = get_legal_moves(state)
        placeable = {m[1] for m in legal if m[0] == "place"}
        t0 = [("place", peg) for peg in KIMI_PRIORITY
              if peg in placeable and len(board[peg]) == 0]
        t1 = [("place", peg) for peg in KIMI_PRIORITY
              if peg in placeable and len(board[peg]) > 0 and board[peg][-1] != p]
        t2 = [("place", peg) for peg in KIMI_PRIORITY
              if peg in placeable and len(board[peg]) > 0 and board[peg][-1] == p]
        t3 = [m for m in legal if m[0] == "remove"]
        return [t0, t1, t2, t3]

    def _safe(self, state: GameState, move: Move) -> bool:
        """A move is safe iff it doesn't end the game against the mover and
        leaves the opponent without a win-in-1 (one-reply safety horizon).

        Formalization seam: a child that is an immediate terminal loss counts
        as unsafe; an immediate terminal draw counts as safe.
        """
        p = state.turn
        child = apply_move(state, move)
        t = _end_value(child)
        if t is not None:
            return orient(t, p) >= 0
        opp = child.turn
        for om in get_legal_moves(child):
            ot = _end_value(apply_move(child, om))
            if ot is not None and orient(ot, opp) > 0:
                return False
        return True


class KimiAgentV1(_KimiBase):
    """Externally proposed "basic strategy" (Kimi), formalized and FROZEN.

    1. Win-in-1: first legal move (engine order) whose child is a terminal win
       for the mover. Removals are allowed ONLY here (T3 exists as a fallback
       when no placement is legal).
    2. Tiers: T0 place on empty pegs (priority 4,0,2,6,8,1,3,5,7); T1 place on
       opponent-topped stacks; T2 self-stack; T3 removals.
    3. Safety filter WITHIN the first non-empty tier only: drop moves after
       which the opponent has a win-in-1; if all are unsafe, take the tier's
       first move anyway.

    The tier-local safety filter is a formalization seam, and it is
    load-bearing: it never escalates to a blocking move in a later tier, which
    is exactly the hole the certified exploit walks through (see Gate C in
    Finding 8 and ``experiments/best_response.py``). Do not repair; improved
    variants are new named agents.
    """

    def __init__(self, seed: Optional[int] = None):
        self.name = "kimi-v1"

    def choose(self, state: GameState) -> Move:
        w = self._win_in_1(state)
        if w is not None:
            return w
        for tier in self._tiers(state):
            if not tier:
                continue
            for m in tier:
                if self._safe(state, m):
                    return m
            return tier[0]      # all unsafe: take the tier's first anyway
        raise ValueError("choose() called on a state with no legal moves")


class KimiAgentV2(_KimiBase):
    """Second formalization: safety escalates ACROSS tiers. FROZEN.

    Identical to V1 except rule 3: scan all moves in global (tier, priority)
    order and take the FIRST SAFE one anywhere; if nothing is safe, take the
    first move in global order. This is not a repair of V1 -- it moves the
    certified failure deeper (5-ply exploit -> 8-9 plies; see Finding 8).
    """

    def __init__(self, seed: Optional[int] = None):
        self.name = "kimi-v2"

    def choose(self, state: GameState) -> Move:
        w = self._win_in_1(state)
        if w is not None:
            return w
        ordered = [m for tier in self._tiers(state) for m in tier]
        if not ordered:
            raise ValueError("choose() called on a state with no legal moves")
        for m in ordered:
            if self._safe(state, m):
                return m
        return ordered[0]


class HeuristicNPlyAgent(Agent):
    """Depth-limited alpha-beta using the positional ``board_heuristic`` at the
    horizon.

    Unlike :class:`NPlyAgent` -- whose bead-count leaf is provably flat across
    all one-ply children (every legal move shifts the material differential by
    the same +1), making depth-1 equivalent to random play -- this uses a
    *functioning* evaluation. Running it at increasing depth isolates the effect
    of lookahead while holding a working evaluation fixed (see Finding 8).
    """

    _WIN = 1e9  # terminal magnitude, dominates any heuristic leaf value

    def __init__(self, depth: int = 1, seed: Optional[int] = None):
        assert depth >= 1
        self.depth = depth
        self.name = f"heur-{depth}-ply(seed={seed})"
        self.rng = _seeded(seed)

    def _search(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        # All values are from P0's perspective (P0 maximizes, P1 minimizes).
        t = evaluate_terminal(state)
        if t is not None:
            u = orient(t, 0)
            return self._WIN if u > 0 else (-self._WIN if u < 0 else 0.0)
        moves = get_legal_moves(state)
        if not moves:
            u = orient(attrition_value(state.board), 0)  # attrition game-end
            return (self._WIN / 2) if u > 0 else (-(self._WIN / 2) if u < 0 else 0.0)
        if depth == 0:
            return float(board_heuristic(state.board, 0))
        if state.turn == 0:
            best = -2 * self._WIN
            for move in moves:
                best = max(best, self._search(apply_move(state, move), depth - 1, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        best = 2 * self._WIN
        for move in moves:
            best = min(best, self._search(apply_move(state, move), depth - 1, alpha, beta))
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def choose(self, state: GameState) -> Move:
        moves = get_legal_moves(state)
        mover = state.turn
        scored = []
        for move in moves:
            v = self._search(apply_move(state, move), self.depth - 1, -2 * self._WIN, 2 * self._WIN)
            scored.append((orient(v, mover), move))
        best_score = max(s for s, _ in scored)
        best_moves = [m for s, m in scored if s == best_score]
        return self.rng.choice(best_moves)
