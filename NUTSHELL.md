# Collapse3 in a nutshell

Three plain takes:

1. Collapse3 is a tiny, deterministic, fully solved 3×3×3 game in which an exact
   solver grades every move against perfect play. It shows that a policy can
   post an excellent win rate and Elo while still losing to a short, forced
   sequence it never sees coming — and that hiding part of the state creates an
   exact, enumerable performance floor no amount of training can cross. The
   uncomfortable point for machine learning and AI safety: standard evaluation
   missed or badly understated these failures, and they surfaced only because
   exact ground truth existed — the very thing evaluations of real systems
   usually lack.

2. Collapse3 is a small 3×3×3 board game, solved exactly. Because there is ground
   truth for every move, you can grade a proposed *evaluation method* against it
   rather than only grading an agent.

   An agent can look near-perfect under the standard evaluation (play many
   games, count wins) while having a fatal flaw that standard evaluation
   structurally cannot see — but an exact solver finds instantly.

3. Collapse3 shows that a game can be easy to solve with a computer,
   surprisingly hard to compress into rules a human can hold, and dangerously
   deceptive to a pattern-matching machine — all at once. And because the game
   is solved, none of it is opinion.

Full write-up: [`README.md`](README.md) and [`docs/FINDINGS.md`](docs/FINDINGS.md).
