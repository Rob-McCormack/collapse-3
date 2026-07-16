# Collapse3

<img src="docs/images/3x3-simple.png" width="640" alt="Collapse3 board: a 3×3 grid of pegs holding red and blue beads">


**Is competence a property of the agent — or of the agent plus the opponents it
happens to face?**

A one-ply agent posts its highest optimal-move rate (**78%**) against the
opponent it loses to **84%** of the time (and wins **0%**). In a measured
round-robin, **Elo ranks a certifiably exploitable rulebook above the perfect
player** ([Finding 10](docs/FINDINGS.md)). Collapse3 is a
**perfect-information, deterministic** game small enough to solve exactly — and
the **exact solver is included in this repository** (pure Python, no
dependencies). The solver isn't a player here, it's an examiner: it grades every
move against perfect play, turning "how good is this agent?" into a number —
**value-based regret**. No opponent can make a bad move grade as good, and no
thrown game grades clean — surrendered value always logs as regret
([Finding 9](docs/FINDINGS.md)). What an opponent *can* still shape is which
states get visited, which is why every average we report names its
distribution.

In large systems we watch agents fail and wonder why. In Collapse3, we can
measure *exactly why* — and **unit-test proposed evaluation methods against
ground truth** before trusting them on systems you cannot solve. It aims to be
a small, fully reproducible exhibit of the failure modes — distribution-dependent
competence, representation gaps, brittle plans, sandbagging — that AI safety and
evaluation researchers care about, measured exactly rather than estimated.

*New to the game? The rules fit in ten lines — see [The game](#the-game) below
(full rules: [`rules.md`](rules.md)). Enough to read the findings: players
**place** beads from a limited **reserve** onto a 3×3 grid of 3-deep pegs, or
**remove** an opponent bead — destroying it and dropping everything above it —
to make three-in-a-row anywhere in the cube.*

## The result that motivates this

From a **drawn** position, a *frozen* exact-oracle plan loses **15 of 32**
opponent blunder-lines (and wins 0); a player that **re-solves** each move loses
0 and wins 23. A deliberately *worse* move beats the oracle-derived plan —
because the plan was a best-response to one line, not a strategy. Win rate hides
this completely.

## A "basic strategy" exists at the smallest size — and is certifiably dead beyond it

*A compact set of rules you can teach provably runs out here: it works at the
smallest board and is exactly refuted beyond it, so competence becomes a
calculation, not a rule.*

Most games have a *strategy ladder* — a simple positional heuristic gets a
beginner to coherent play. In Collapse3 the ladder has exactly one known rung.
The *intuitive* rule ("build your own lines") is optimal on **0%** of the
decisive moves and loses **99%** of a game it should draw — gravity-and-removal
turns greed into suicide. Even a one-ply agent with a *good* positional
evaluation still **loses 64%** of that drawn game; a single extra ply lifts the
same evaluation to near-perfect (critical-decision accuracy **0.62 → 0.997**).

Then someone proposed an actual rulebook, and we could *prove* things about it.
An externally proposed five-line strategy (place centre → corners → edges,
block, remove only to win) is a **certified exact drawing policy at (3,3)** —
both seats, against every possible opponent, by exhaustive best-response solve.
From **(4,4) up, every formalization of it is a certified forced loss** (both
seats). The teachable game is real, and it ends at (3,3): beyond that, for
every rulebook tested so far, competence is bought in plies of search, not in
rules you can write down — a conjecture with two certified refutations behind
it, not a theorem, and the best-response solver
([`experiments/best_response.py`](experiments/best_response.py)) will grade any
new candidate exactly.

The evaluation moral is the sharpest part: that same rulebook survived **1,199
of 1,200 games** against a strong noisy opponent — while carrying a **5-ply
forced refutation** the exact solve finds in ~0.03 seconds. Playing lots of
games, the way most agents are evaluated, missed the shallowest possible kill
([Finding 8](docs/FINDINGS.md)).

## Competence has a price, and it scales

Exact regret floor of the best memoryless policy that sees a lossy observation
— and *only* the observation, not the state's legal-move list (win/draw/loss
units), by game size. *Reserves* = the beads each player starts with (14 in the
full game); smaller counts `(r, r)` keep the game exactly enumerable, and larger
reserves mean more material and longer games.

| reserves | states | hide cooldown | hide reserves |
|---|---|---|---|
| (2,2) | 4K | 0.0000 | 0.0000 |
| (3,3) | 97K | 0.0003 | 0.0028 |
| (4,4) | 1.36M | 0.0024 | 0.0805 |
| (5,5) | 12.7M | 0.0034 | **0.1677** |

The cost grows with the game — it is **structural, not small-game triviality**.
More sophisticated training cannot help, because the failure isn't in the
optimizer: the distinguishing information is absent from the input — and every
method that appears to help works by putting it back. That includes subtle
routes: give the policy the **legal-move list** and the mask itself leaks state
(removals reveal cooldown; missing placements reveal an empty reserve) — the
cooldown floor drops to (nearly) 0 and the reserves floor collapses ~30–70×
(0.0805 → 0.0026 at (4,4); 0.1677 → 0.0024 at (5,5)). The floor is a property
of the *interface*, priced exactly per interface (Finding 4).

**The quiet twist: a floor is a property of an *interface*, not a game or an
agent.** The **same** missing feature (the reserve count) costs a ladder of
different exact amounts depending on what the policy sees: **0.0805** to a
(board, cooldown) agent, **~0.0026** once it also holds the legal-move list (the
interface our trained agents actually had — the mask leaks "zero vs. positive
reserve"), and **0.0000** with destroyed-bead memory. On top of that, *realized*
on-policy regret is a different, opponent-dependent quantity again (~0.0013 vs an
ε-optimal opponent, ~0.24 vs a random one). We priced the first rung, our agents
lived on the second, and only measuring all of them showed which number was
honest — the exact failure mode of real evaluation: *the bound you prove is for
the system you modeled, not the one you shipped* (Findings 4 and 7).

Collapse3 is an exact, game-based demonstration of the memoryless-policy problem
Littman (1994) formalized. That such policies pay a floor under aliased
observations is classical (Whitehead & Ballard 1991; Littman 1994; Singh,
Jaakkola & Jordan 1994); what Collapse3 adds is the *exact*, enumerated value of
that floor in a natural, fully solved game — not a worst-case bound — its growth
with size, and a constructive recovery to a **0.0000** floor from remembered game
history ([`docs/FINDINGS.md`](docs/FINDINGS.md)).

Collapse3 is a deterministic, perfect-information game turned into an instrument
for studying partially observable decision-making: restricting what an agent
observes induces a POMDP for that agent — but inside a fully solved game, so the
evaluator keeps exact ground-truth values and every decision is graded. The cost
of a missing feature becomes a number you *enumerate*, not one you estimate.

The game gets *sharper* as it grows, not drawish: the drawn fraction falls
**86% → 72% → 54% → 38%** from (2,2) to (5,5), and equal reserves — a draw
through (5,5) — become a **first-player line win at (6,6)**, extending through
the full **14-bead game** (Finding 11). Scale the reserves, not the rules, if
you want a less drawish game. Move order and material decide the opening
exactly, which is why win rate needs the oracle to deconvolve it. Details in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## Throwing the game is provably hard

Deliberate underperformance is structurally resisted here. We solved the misère
question exactly: a player who *wants* to lose **cannot force the opponent to
win** at any solved size ((3,3)–(5,5)), from either seat — he can force never
*winning* (accept a draw), but the opponent can always escape a forced loss.
Against a random opponent a perfect thrower can raise the opponent's win chance
steeply with size (**31% → 55% → 80%** moving first) — but never to 100%:
forcing your own loss is impossible at every solved size. What a regret audit
sees is two-sided, and exactly measurable: *realizing* a worse outcome than the
position warrants always logs positive regret, but a thrower can **steer inside
the set of optimal moves for free** — at (3,3) all nine openings are zero-regret,
yet they differ in how likely the fallible opponent is to end up winning, and
the oracle cannot see that choice. The oracle audits thrown *value*, not thrown
*intent*. Or, concretely: a grandfather provably cannot guarantee his grandson
wins ([Finding 9](docs/FINDINGS.md)).

## A self-play agent looks flawless — and walks off a cliff

*The same evaluation blind spot, now with a learned agent that never saw the
solver.* On an exactly-solvable **sibling** of the game (three pegs; a parallel
environment, never a claim about Collapse3), a **true self-play** agent — one
shared brain playing both seats, learning only from who won — becomes
near-perfect **on its own trajectories** (~0.045 regret) and *stays* there as
the game grows. Graded over the *whole* game by the exact solver, its regret
**triples** (0.06 → 0.21) and it has only ever visited **42%** of the positions:
brilliant where it plays, blind everywhere else. Freezing that "undefeated"
policy, an exact best-response forces it off its winnable game in **2 of 5**
runs at the largest size — a flat self-play record is not robustness. And the
sharpest failure is distribution shift: trained where the centre opening
*uniquely wins*, **all five** runs learn "play centre" and carry it across the
enumerated phase boundary into a size where the centre **loses** — provably
optimal in training, fatal one reserve later ([Finding 14](docs/FINDINGS.md)).

## The game

![Ready, fire, aim](docs/images/ready-fire-aim.png)

Ready, fire, aim. Most games are ready, *aim*, fire. Collapse3 is ready, *fire*,
aim: you commit a bead before the board has decided what it's worth. Pull a bead
out and everything above it drops a level — so a piece you placed earlier can
fall into a different line, or out of the one you meant. And the only beads you
can pull are your opponent's: you play inside their position, not just your own.

**Rules in ten lines.** The board is a 3×3 grid of pegs, each holding up to 3
beads; each player starts with a **reserve** of beads (14 in the full game;
`(r, r)` in the experiments). On your turn you do exactly one thing:

- **Place** a bead from your reserve on any non-full peg — it falls to the
  lowest empty level; or
- **Remove** one *opponent* bead ("the Collapse") — legal only from a peg that
  is tied-tallest, holds 2+ beads, and is *opponent-topped*, and never on two
  of your consecutive turns (**cooldown**). The bead is destroyed — it returns
  to no one — and everything above it drops one level (**gravity**).

First to line up three of their beads wins — vertical on one peg, flat on any
level, or a staircase across three collinear pegs. If your removal's gravity
cascade completes a line for your *opponent*, they win instantly (the **Oops
rule**). When both reserves are empty, or the player to move has no legal
action, the game ends immediately and most surviving beads on the board wins
(**attrition**); equal count is a draw.

So the top of the board isn't the whole story: buried beads, reserve counts, and
last turn's cooldown decide the game too. Even chess hides a little history in
its state — castling rights, en passant, the fifty-move counter — none of it
visible in a bare snapshot; Collapse3 just makes that gap **load-bearing**, which
is exactly why a lossy view of the state carries an irreducible cost (see the
floor above). Full rules (singleton immunity, simultaneous wins, end-of-game
priority) in [`rules.md`](rules.md).

## More

- **Findings & significance:** [`docs/FINDINGS.md`](docs/FINDINGS.md)
- **FAQ** (is this a POMDP? is (6,6) a bug? would a bigger model help? why not
  Elo?): [`docs/FAQ.md`](docs/FAQ.md)
- **Perspective** (optional interpretation, not results): [`docs/PERSPECTIVE.md`](docs/PERSPECTIVE.md)
- **Ask an LLM about this repo:** paste [`llms-full.txt`](llms-full.txt) (the
  entire project in one ~108K-token file) into any chat model. Index: [`llms.txt`](llms.txt).

## Reproduce

```bash
pip install -e ".[dev]" && pytest
python -m experiments.aliasing_floor 4        # any experiment writes results/<name>.json
pytest tests/test_reference_engine.py       # clean-room rules cross-check (~5s)
```

```
collapse3/   engine + solver + oracle + agents + learning + metrics
experiments/ provenance-stamped studies (python -m experiments.<name>)
results/     outputs with the git commit, config, and seed that produced them
```

## Citation

If you use Collapse3, please cite it:

```bibtex
@misc{mccormack2026collapse3,
  author       = {McCormack, Rob},
  title        = {Collapse3: measuring competence vs. performance with an exact game oracle},
  year         = {2026},
  howpublished = {\url{https://github.com/Rob-McCormack/collapse-3}}
}
```
