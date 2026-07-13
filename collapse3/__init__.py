"""Collapse3: exact solver and competence-measurement toolkit.

Submodules:
    game     -- pure rules engine (state, moves, win/terminal detection)
    solver   -- exact minimax + alpha-beta + transposition table
    oracle   -- ground-truth value labels and value-based regret
    agents   -- reference policies (optimal, n-ply, random, myopic)
    metrics  -- competence measurement (regret distributions, criticality)
"""

from . import game  # noqa: F401

__version__ = "0.2.0"
