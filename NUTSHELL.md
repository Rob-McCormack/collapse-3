# Collapse3 in a nutshell

> Collapse3 is a tiny, deterministic, fully solved 3×3×3 game in which an exact
> solver grades every move against perfect play. It shows that a policy can post
> an excellent win rate and Elo while still losing to a short, forced sequence it
> never sees coming — and that hiding part of the state creates an exact,
> enumerable performance floor no amount of training can cross. The uncomfortable
> point for machine learning and AI safety: standard evaluation missed or badly
> understated these failures, and they surfaced only because exact ground truth
> existed — the very thing evaluations of real systems usually lack.

---

## What it is

Collapse3 is a small, deterministic 3×3×3 game with an exact solver that grades
every move against perfect play. The findings: win rate and Elo can hide short,
forced losses even when average accuracy looks near-perfect; missing information
creates exact, enumerable performance floors; and simple rulebooks that hold up
at the smallest size break under exact adversarial play as the game grows. (In
one measured tournament, Elo even ranks a *certifiably exploitable* rulebook
above the *perfect* player.)

## How the game works

Two players take turns dropping beads onto nine pegs, each three levels high,
trying to line up three of their own — flat, vertical, or as a staircase across
levels. Instead of placing, a player may sometimes remove one of the opponent's
beads from an eligible tallest stack; everything above it drops a level and the
removed bead is destroyed for good. If neither side has made a line by the time
the beads run out (or a player has no legal move), whoever has more beads left
on the board wins; an equal count is a draw. Full rules: [`rules.md`](rules.md).

## Why it matters

For AI safety and evaluation the point is methodological: in a fully solved
setting, strong sampled performance is demonstrably not the same as robust
competence. Because the ground truth is exact, Collapse3 lets researchers
measure distribution shift, brittle policies, representation failures, and
adversarial exploitability without having to guess what the correct answer was.

Full write-up: [`README.md`](README.md) and [`docs/FINDINGS.md`](docs/FINDINGS.md).
