# Findings & Significance

All numbers below are produced by the scripts in `experiments/` and written to
`results/*.json` with the git commit, config, and seed that generated them.
Figures are for the shipped rule (**immediate game-end on no legal action**);
exhaustive results are exact, and any illustrative small-budget run is labelled.
The one exception is **Finding 16**, a torch-dependent *exhibit* quarantined in
`probes/`: its 12/12 forced-loss outcome is reproducible, but neural decimals
drift with the torch build (see [`docs/NEURAL_EXHIBIT.md`](NEURAL_EXHIBIT.md)).

---

## Findings at a glance

*One line per finding — click through to its full form below.*

1. **[Aggregate metrics are adversarially flattering](#1-aggregate-metrics-are-adversarially-flattering)** — most states have many optimal moves, so headline "optimal-move rate" hides the decisive mistakes.
2. **[Performance and competence come apart](#2-performance-and-competence-come-apart)** — win rate and value-based regret diverge; the same agent looks strong or weak purely by opponent.
3. **[An open-loop plan is not a strategy](#3-an-open-loop-plan-is-not-a-strategy--shown-exactly)** — a frozen oracle-derived plan is a best-response to one line; a re-solving player crushes it.
4. **[Representation cost is real, quantifiable, and interface-dependent](#4-representation-cost-is-real-quantifiable-and-interface-dependent)** — exact floors per interface; the sharpest result is *opposite scaling signs*: across the two enumerable sizes the mask-blind hide-reserves floor rises (0.08 → 0.17) while the shipped mask-aware floor declines (0.0026 → 0.0024).
5. **[Objective failure vs. representation failure](#5-objective-failure-vs-representation-failure)** — two distinct failure modes separated exactly: missing information vs. the wrong objective.
6. **[Seat and material decide the opening](#6-seat-and-material-decide-the-opening--a-hidden-confound-in-win-rate)** — opening value is fixed by seat and reserves, a hidden confound baked into any win-rate comparison.
7. **[The floor is real, but realized regret slips beneath it](#7-the-floor-is-real-but-realized-regret-slips-beneath-it--interface-first-then-steering)** — the agent's true interface erases most of the aliasing floor and its trajectories steer around the rest.
8. **[Is there a teachable "basic strategy"?](#8-is-there-a-teachable-basic-strategy-certified-yes-at-33--certified-refutations-from-44-up)** — a compact rulebook is a certified draw at (3,3) and a certified forced loss from (4,4) up.
9. **[Throwing the game is provably hard](#9-throwing-the-game-is-provably-hard)** — you cannot force the opponent to win at any solved size; the oracle audits thrown value, not intent.
10. **[Elo prefers the exploitable agent](#10-elo-prefers-the-exploitable-agent--the-rating-inversion-measured)** — in a round-robin, Elo rates a certifiably exploitable rulebook above the perfect player.
11. **[The full game fell — (14,14) is a first-player forced win](#11-the-full-game-fell--1414-is-a-first-player-forced-win-found-by-accident)** — the full game is a first-player win, found by accident; the drawish picture flips with capacity.
12. **[A sibling game shows the floor's *shape*](#12-a-sibling-game-shows-the-floors-shape--bounded-growth-and-a-hidden-feature-whose-cost-rises-then-falls)** — a three-peg sibling maps the floor across all sizes: steep then flattens near 0.25 over the game-range; cooldown rises then falls.
13. **[A second sibling shows the boundary *moves* with capacity](#13-a-second-sibling-six-pegs-shows-the-boundary-moves-with-capacity--finding-12iv-tested)** — a six-peg sibling confirms the phase boundary shifts later as board capacity grows (Finding 12(iv), tested).
14. **[Self-play saturates: flawless where it plays, weaker off its lines, exploitable, and it trips the boundary](#14-self-play-saturates-flawless-on-its-own-trajectories-weaker-off-them-and-it-trips-over-the-phase-boundary)** — true self-play looks perfect on its own games yet its global edge over a random baseline *erodes* with size, it is exactly exploitable, and it is forced to lose when it carries a trained-optimal opening across the phase boundary. Part (iv): the exploiter need not be strong — a policy random except on six memorized moves beats a >99%-vs-random champion (KataGo analogue).
15. **[Knowing when to look — the value of information is sparse](#15-knowing-when-to-look--the-exact-value-of-information-is-sparse)** — an exact information policy: missing ≠ useful (reserves have zero decision value at (3,3)), yet from (4,4) up only ~1% of decisions must inspect reserves and that 1% carries the *entire* irreducible floor.
16. **[Generalization is not robustness](#16-generalization-is-not-robustness--a-trained-net-that-generalizes-is-still-certifiably-force-losable)** — a small net that plays ~98% optimally on unseen (4,4) states is still a certified forced loss from both seats; all six trained nets fall to the exact adversary from both seats (12 of 12 certifications). *(Torch-dependent exhibit — see [`docs/NEURAL_EXHIBIT.md`](NEURAL_EXHIBIT.md).)*
17. **[Passing a test rules out less than you think](#17-passing-a-test-rules-out-less-than-you-think--the-strongest-player-is-not-the-strongest-tester)** — the exact *compatible outcome range* of an evaluation: an evaluator that tests only against **optimal** opponents leaves a **certified forced loss** compatible (as few as **10** pinned decisions at (4,4), two-sided), while all-legal coverage rules it out. Strategic coverage buys nothing over same-universe random — it's the universe, not the selection (H2 falsified). *(See [`docs/EVALUATION_EQUIVALENCE.md`](EVALUATION_EQUIVALENCE.md).)*

**Also in this document:** [The measurement problem](#the-measurement-problem) · [Where the difficulty lives](#where-the-difficulty-lives) · [Relation to prior work](#relation-to-prior-work) · [Why this matters for AI research](#why-this-matters-for-ai-research) · [Is this just undertraining?](#is-this-just-undertraining-would-a-bigger-model-help) · [Rule sensitivity](#rule-sensitivity-a-caution) · [Limitations & scaling](#limitations--scaling) · [Reproducibility](#reproducibility)

---

## The measurement problem

Standard evaluation reports **performance** (win rate against some opponent
pool). But win rate is a convolution of two things — the agent's skill and the
opponent's policy — and it never deconvolves them. An agent can look strong,
brittle, or worthless purely because of *who it played*.

Collapse3 is small enough to **solve exactly**, which lets us measure the other
quantity directly: **competence**, defined as **value-based regret** against the
oracle. A move is optimal iff it preserves the state's game-theoretic value from
the mover's seat; its regret is the value it gives up. The per-decision *label*
is opponent-independent — no weak opponent can make a bad move grade as good.
But any **average** of those labels is taken over the states actually visited,
which the opponent shapes — the same agent scores 0.0019 against one opponent
distribution and 0.0508 against another (Finding 5). So we report regret
*per distribution* and name the distribution, rather than pretending a single
scalar transfers. For *realized* (learned-policy) regrets we also print a
**random-policy baseline** on the same denominator, so a number like "0.215"
has a scale and coverage is never blended into "competence" (Finding 14).
Exact interface floors already *are* the optimum over memoryless policies — they
need the interface named (the ladder below), not a random baseline.

We deliberately use *value* regret, not policy distance `|π_agent − π_opt|`:
optimal play here is highly non-unique (many moves hold a draw), so policy
distance would wrongly penalize correct-but-different moves.

### A representation floor is a property of an *interface*, not a game or an agent
`experiments/interface_ladder.py` (reserves (4,4))

Before the floor tables below, the single measurement lesson they add up to. A
"representation cost" is meaningless until you say *from which interface*. For the
**same** missing feature — the reserve count, hidden under `hide_reserves` — the
exact cost is a ladder, and every rung is a different, exactly-enumerated number:

| what the policy sees | (4,4) cost | kind |
|---|---|---|
| board, turn, cooldown (reserve fully aliased) | **0.0805** | exact interface floor |
| + the legal-move list (what our trained agents actually saw) | **0.0026** | exact interface floor |
| + destroyed-bead memory (reserves reconstructed) | **0.0000** | exact interface floor |
| ⟶ *its own on-policy trajectory* (trained, ε-optimal opponent) | ~0.0013 | **not a floor** — see below |

**Same missing feature, opposite scaling signs.** Across the two full-board sizes
we can enumerate, the mask-blind reserve floor **grows** (0.0805 → **0.1677**)
while the mask-aware floor — the interface our trained agents actually had —
**declines** slightly (0.0026 → **0.0024**; Finding 4 / `masked_floor.py`).
"Grows with size" is true of the theoretical anchor, not of the shipped
interface. Pricing only one rung would have told the wrong scaling story.

The first three rungs are exact, uniform-over-decision-state floors (the Finding 4
machinery) and are **opponent-independent**. The mask rung is load-bearing and easy
to miss: our `QAgent` keys its table on the lossy observation but argmaxes over
`get_legal_moves`, so it *always* holds the legal-move list — and that mask leaks
"zero vs. positive reserve," which is why the same feature costs **30×** less to
the agent we actually trained (0.0805 → 0.0026) than to the mask-blind interface we
first priced. Memory erases the rest.

The fourth line is a **different quantity**: realized on-policy regret, which is
trajectory- **and** opponent-dependent — ~0.0013 against an ε-optimal opponent but
~**0.24** against a random one (Finding 7). It is listed here only to keep it from
being confused with a floor; it belongs to a trajectory, not an interface.

This is the sharpest measurement statement in the project, and it is the exact
failure mode of real evaluation: **the bound you prove is for the system you
modeled, not the system you shipped.** We computed the first rung, our agents lived
on the second, their behavior realized the fourth, and only measuring all of them
revealed which number was the honest one. (Findings 4 and 7 are the two halves;
this is their synthesis.)

---

## What we demonstrate

### 1. Aggregate metrics are adversarially flattering
`experiments/optimal_move_rate.py` (reserves (4,4))

A one-ply searcher's optimal-move rate, overall vs. on **critical** decisions
(states where a mistake is even possible):

| opponent | win rate | optimal-rate (all) | optimal-rate (critical) |
|---|---|---|---|
| random | 0.75 | 0.711 | 0.531 |
| optimal | 0.00 | **0.783** | **0.359** |

The aggregate rate is highest (0.78) against the opponent it **loses to 84% of
the time**. Difficulty concentrates in a small set of critical states; any
metric that doesn't condition on decision-criticality overstates competence.

The rating-system version of the same failure is Finding 10: fitted **Elo
ranks a certifiably exploitable rulebook above the perfect player**, because
ratings reward converting wins against the weak, not distance from ground
truth.

### 2. Performance and competence come apart
`experiments/oneply_vs_learned.py` (reserves (4,4), 150K episodes/seat)

> **Correction (external review).** An earlier version of this finding claimed
> the learned agent "loses every game as second player." That number was an
> artifact: the Q-table was trained at seat 0 only (its observation key includes
> the turn), so played as P1 it had no trained entries and fell back to
> first-legal-move — an untrained policy, not a trained agent off-distribution.
> The experiment now trains one policy per seat; the numbers below are the
> corrected, seat-matched results.

Audited against the oracle while playing an eps-optimal opponent, the learned
tabular agents are near-optimal in both seats — mean regret **0.0019** at seat 0
(0 losses in 2000 games) and **0.0191** at seat 1 — far better than the one-ply
heuristic's **0.1726**. Yet head-to-head, the seat-1 learner **loses 29%** of
games (58/200, with 123 draws) to the one-ply agent — an opponent whose own
audited regret is ~9× worse — because the one-ply agent's unfamiliar lines
visit states outside the learner's training distribution, where its Q-values
are unreliable. (First-move advantage is strong: the seat-0 learner beats the
one-ply agent 200/200.) A near-optimal *average* regret against one opponent
distribution does not transfer to another: the average is taken over the states
that opponent induces, and a different opponent induces different states. The
per-decision oracle labels are opponent-independent; a trajectory-averaged
competence score is not.

### 3. An open-loop plan is not a strategy — shown exactly
`experiments/frozen_plan_vs_blunder.py` (reserves (4,4))

Starting from a **drawn** root, we enumerate every single-deviation move by the
opponent (23 strictly value-losing blunders + 9 alternative value-optimal moves).
Against a player replaying a **frozen** principal-line plan:

- static frozen plan: **0 wins, 17 draws, 15 losses**
- re-solving each move: **23 wins, 9 draws, 0 losses**

15 **inversions**: a deliberately *worse* opponent move beats the exact
oracle-derived plan. Framed precisely: a principal variation is an **open-loop
action sequence**, not a contingent strategy — it carries no branch for states
off its line, so replaying it against deviations exercises undefined behaviour,
and re-solving (or an equivalent precomputed *policy table*) is the closed-loop
fix. That distinction is elementary; the value here is pedagogical and
quantitative, not conceptual novelty — the game lets us enumerate *every*
deviation and show the failure is 15/23 of strict blunders, not an edge case.
It is a clean, exact miniature of the open-loop failure mode that recurs in
plan-then-execute agent designs.

### 4. Representation cost is real, quantifiable, and interface-dependent
`experiments/aliasing_floor.py` (exhaustive, reserves (2,2)–(5,5))

The regret floor of the best *memoryless* policy over a defined **interface**:
the policy sees only the lossy observation — *not* the state's legal-move list —
and must commit to one action per observation (win/draw/loss units, uniform
weighting over reachable decision states). *Reserves* = the beads each player
starts with; `(r, r)` is the symmetric starting count, and larger reserves mean
more material and a longer game. The interface qualifier matters — see "The
interface is part of the claim" below.

| reserves | states | hide cooldown | hide reserves | hide both |
|---|---|---|---|---|
| (2,2) | 4,051 | 0.0000 | 0.0000 | 0.0000 |
| (3,3) | 97,093 | 0.0003 | 0.0028 | 0.0003 |
| (4,4) | 1,357,963 | 0.0024 | 0.0805 | 0.0003 |
| (5,5) | 12,714,999 | 0.0034 | **0.1677** | 0.0004 |

Across the two full-board sizes we can enumerate, the reserves-hidden floor
*rises* (≈2× from (4,4) to (5,5)); the cooldown-hidden floor stays tiny and
plateaus. So the cost of a missing state feature is **structural**, not an
artifact of small-game triviality — and reserves, not cooldown, are the
load-bearing feature. Two full-board points do not by themselves establish a
*trend*, which is why the strongest and most defensible version of the scaling
claim is the **opposite signs** result above (mask-blind rises, mask-aware
falls) — it needs only these two points — and the growth's actual *shape* is
carried by the sibling games (Findings 12–13), which enumerate every size. The
novel content here is not that such a floor *exists* (that is classical — see
[Relation to prior work](#relation-to-prior-work)); it is that we can
**enumerate it exactly** per interface and — for the single-hide columns —
show it is the *exact* optimum, below.

> **Cross-reference — the growth's *shape* is only known for a sibling game.**
> The two points above ((4,4), (5,5)) are the only enumerable full-board data,
> and the growth claim stands exactly as stated *for them*. What they cannot
> reveal is the shape of the curve, because the full board is not enumerable past
> (6,6). [Finding 12](#12-a-sibling-game-shows-the-floors-shape--bounded-growth-and-a-hidden-feature-whose-cost-rises-then-falls)
> enumerates a *sibling* game (Three-Peg Collapse) at every size and finds the
> reserves floor **saturates** (bounded near ~0.25) and the cooldown floor is
> **non-monotonic** — with (4,4)/(5,5) sitting on the steep early section where
> no shape is yet visible. Those are facts about the sibling's geometry, **not**
> Collapse3; the full game's floor shape beyond (5,5) remains unknown.

**Exactness (single-hide columns).** The floor uses a *charity rule*: an
observation group with no action legal in **every** member contributes 0,
keeping the number a valid lower bound. For `hide_cooldown` and `hide_reserves`
that rule **never fires**, so those two columns are not just bounds — they are
the **exact** minimum regret of the best memoryless policy on that observation.
Why every group has a common legal action:

- `hide_cooldown`: members share `(board, turn, reserves)`. Placement legality
  depends only on the board and `reserves > 0` (not cooldown), so if the mover
  can place, the *same* placements are legal in every member; and if it cannot
  place, any member with no legal action is terminal-by-attrition under the repo
  rule and never enters a decision-state group.
- `hide_reserves`: members share `(board, turn, cooldown)`. Removal legality
  depends on board and cooldown (shared), so removals are legal in all members
  or none; when none and a `reserves = 0` member has no placement either, that
  member has no legal action and is excluded — leaving a common placement among
  the `reserves > 0` members.

This is **verified by exhaustive enumeration, not just argued**: the scan
asserts a common legal action exists in every decision-state group
(`no_common_legal_action == 0`) for both single-hide ablations at **(3,3),
(4,4), and (5,5)**. `tests/test_floor_exactness.py` runs the (3,3) scan by
default and the full (4,4) census under `COLLAPSE3_SLOW=1`; the (5,5) scan is
recorded in
[`results/aliasing_floor_latest.json`](../results/aliasing_floor_latest.json)
and a fast test asserts that shipped value. That last point is load-bearing: the
headline **0.1677** floor lives at (5,5), so *its* exactness rests on the (5,5)
enumeration, not on the proof sketch. The sketch above explains *why* every
group has a common legal action and is expected to hold at all sizes, but the
guarantee at any given size is the enumeration at that size. `hide_both` is the
deliberate contrast, where the rule fires on hundreds of thousands of groups at
(5,5).

> **Footnote — the `hide_both` column is a deflated lower bound.** Hiding
> reserves *and* cooldown makes action **legality itself** diverge across aliased
> states, so the charity rule fires on most conflict groups (36,537 of them at
> (4,4)) and zeroes their contribution — which is why `hide_both` (0.0003) comes
> out *smaller* than `hide_reserves` (0.0805), something a true cost measure
> could never do. The non-monotonicity is an artifact of the convention, **not a
> property of the game** (see the Exactness note above); it is a bound only, not
> comparable to the exact single-hide columns, and is shown for completeness. The
> load-bearing, strictly monotonic comparison is `hide_cooldown` vs
> `hide_reserves`.

> **Aliasing, not secrecy.** Collapse3 is perfect-information — reserves and
> cooldown are on the table. The lossy learner simply isn't *fed* them. The cost
> above is aliasing (two different states sharing one observation), not hidden
> information.

**The interface is part of the claim (external review; `experiments/masked_floor.py`).**
The table above prices a **mask-blind** interface. The trained `QAgent` in
Findings 5 and 7 has a *richer* one: it receives the state's **legal-move list**,
and that mask leaks hidden state — under `hide_cooldown`, seeing removal moves
in the list reveals the cooldown bit; under `hide_reserves`, seeing no
placements reveals an empty reserve. Recomputing the exact floors with the
legal-action mask folded into the observation:

| reserves | interface | hide cooldown | hide reserves | hide both |
|---|---|---|---|---|
| (3,3) | mask-blind (table above) | 0.0003 | 0.0028 | 0.0003* |
| (3,3) | mask-aware | **0.0000** | **0.0000** | 0.0000 |
| (4,4) | mask-blind (table above) | 0.0024 | 0.0805 | 0.0003* |
| (4,4) | mask-aware | **0.0000** | **0.0026** | 0.0030 |
| (5,5) | mask-blind (table above) | 0.0034 | 0.1677 | 0.0004* |
| (5,5) | mask-aware | **0.00001** | **0.0024** | 0.0030 |

\* deflated lower bound (charity rule, footnote above). The mask-aware floors
are exact optima with **no** charity deflation (every masked group shares one
legal-move set, so the rule cannot fire) — note mask-aware `hide_both` ≥
`hide_reserves`, restoring the monotonicity the mask-blind column lacks.

Three consequences, stated precisely. (i) The **cooldown floor is almost
entirely mask-leakage**: the *mover's* cooldown acts only through removal
legality, so the legal-move list usually reconstructs it — the mask-aware floor
is exactly 0 at (3,3)/(4,4), with a trace residual at (5,5) (0.000012, from
cooldown information the mask cannot carry: the opponent's flag, and the
mover's own flag when no removal is available regardless). (ii) The **reserves
floor survives the mask but collapses ~30–70×** (0.0805 → 0.0026 at (4,4);
0.1677 → 0.0024 at (5,5)): the mask reveals *zero vs. positive* reserve but not
the count, leaving a real residual. And the two interfaces scale with **opposite
signs**: mask-blind hide_reserves **grows** 0.0805 → 0.1677, while mask-aware
**declines** 0.0026 → 0.0024. "The floor grows with size" is true of the
theoretical anchor, not of the interface the repo's agents shipped under.
(iii) The headline numbers (0.0805, 0.1677) are therefore **exact for the
mask-blind interface, not lower bounds on the repo's own trained agents**, whose
interface is mask-aware. The honest one-line reading: *how much a missing feature
costs depends on the interface it is missing from — and Collapse3 can price each
interface exactly.* The full ladder for the reserve feature (mask-blind →
mask-aware → +memory, plus the separate on-policy layer) is collected in the
measurement-problem preamble and is the synthesis of this finding with Finding 7.

### 5. Objective failure vs. representation failure
`experiments/objective_failure.py` (reserves (4,4), 150K episodes)

Not every shortfall is a representation problem. The **same** full-state learner
(representation sufficient), same algorithm — trained only against different
opponents:

| trained against | win vs random | regret vs optimal | regret off-dist (vs random) | losses vs optimal |
|---|---|---|---|---|
| random opponent | 0.984 | 0.0581 | 0.2505 | 31 / 2000 |
| optimal opponent | 0.895 | **0.0019** | 0.0508 | 0 / 2000 |

Trained against a feeble opponent the agent wins ~98% yet carries real regret —
terminal reward gives almost no gradient toward optimal play. Train against a
punishing opponent and the regret nearly vanishes. This **objective** failure is
fixable by a stronger signal — unlike the **representation** floor (Finding 4),
which no opponent or training budget can cross.

**Off-distribution probe.** The last column re-grades each agent on the
trajectories a *random* opponent induces — states it rarely saw while training.
The vs-optimal agent's regret jumps from **0.0019 to 0.0508 (≈27×)** off its
training distribution: near-optimal competence measured on-distribution is *not*
a scalar property that transfers. Even a correctly-represented, near-perfect
agent is only near-perfect *where it was trained* — the same distributional-shift
failure as the frozen plan (Finding 3), now for a learned policy.

### 6. Seat and material decide the opening — a hidden confound in win rate
`experiments/opening_values.py` (exact opening values, reserve grid to (6,6))

The exact value of the empty board, swept over reserve splits `(r0, r1)` with P0
to move (`+` favors the mover, `L`=line win, `a`=attrition win, `..`=draw):

|      | r1=1 | r1=2 | r1=3 | r1=4 | r1=5 | r1=6 |
|------|------|------|------|------|------|------|
| r0=1 | `..` | `..` | `..` | `..` | `..` | `..` |
| r0=2 | `a+` | `..` | `..` | `..` | `..` | `..` |
| r0=3 | `a+` | `a+` | `..` | `..` | `..` | `..` |
| r0=4 | `a+` | `a+` | `a+` | `..` | `..` | `..` |
| r0=5 | `a+` | `a+` | `a+` | `a+` | `..` | `..` |
| r0=6 | `a+` | `a+` | `L+` | `L+` | `L+` | `L+` |

Three exact surprises:

- **First player never loses; second player never wins.** Everywhere in the grid
  the value is `>= draw` for the mover — even when the *replier* holds up to +6
  reserves (verified out to `(1,7)`). A material deficit for the mover is fully
  repaid by tempo; a material lead for the replier buys only a draw.
- **A reserve lead pays off only for the mover.** P0 wins iff it has *strictly
  more* reserves (lower triangle); equal or fewer is a draw.
- **A phase transition with material.** The equal-reserve diagonal is a draw for
  r = 1..5, then flips to a first-player **line win at (6,6)**. The *fair* game
  is decisive once there is enough material — and the win type escalates
  attrition → line as the mover's edge grows. (Cross-checked folded-solver vs.
  independent enumeration on every feasible size.)

**A sharper game without rule changes — just scale the reserves.** The fraction
of reachable states that are **drawn** falls **strictly monotonically** as
material increases (exact enumeration):

| reserves | states | draw % | decisive % |
|---|---|---|---|
| (2,2) | 4K | **85.8%** | 14.2% |
| (3,3) | 97K | **71.5%** | 28.5% |
| (4,4) | 1.36M | **54.0%** | 46.0% |
| (5,5) | 12.7M | **37.7%** | 62.3% |

At `(6,6)` exact enumeration is infeasible, but the **opening** is already a
first-player line win — the fair game becomes decisive before the full state
space can be counted. More material means fewer draws and sharper outcomes; no
rule change required. This is one knob for a "less drawish" research game: scale
`(r, r)` upward rather than redesign the mechanics.

Why it matters here: **move order is a confound in win rate, exactly like
opponent policy.** Two equally competent agents can post opposite records purely
from seat assignment — which is why Finding 2 reports head-to-head *by seat*, and
why the oracle's per-decision regret (seat- and opponent-independent) is the
honest competence signal.

### 7. The floor is real, but realized regret slips beneath it — interface first, then steering
`experiments/memory_floor.py` (reserves (4,4) and (5,5); seeds 0–2 where noted)

Memory vs. memorylessness, separated cleanly. **How to read the trained numbers:**

- A gap between blind and full that **shrinks from 300K → 600K** episodes is
  **coverage**, not aliasing.
- A gap that **persists across budgets** while memory tracks full is the aliasing
  separation we would test for.
- Regret paid **equally by the full-state agent** is coverage/undertraining by
  definition — do not attribute it to hidden information.
- Orderings within **~0.001** are seed noise; do not treat them as findings.

**Enumeration (exact, unconditional).** At (4,4): memoryless hide-reserves floor
**0.0805** (all decisions), memory-augmented **0.0000**. At (5,5): **12,714,999**
reachable states, hide-reserves floor **0.1677** (all decisions), memory-augmented
**0.0000**. Board + cooldown + destroyed-bead counts (from observed removals)
reconstructs reserves as `r − on_board − destroyed`.

**(4,4), 150K episodes, single seed — baseline**

| observation | enumerated floor | regret (eps-optimal) | regret (random opp.) |
|---|---|---|---|
| full | — | 0.0019 | 0.2505 |
| hide reserves | 0.0766 (0.0805 all) | 0.0013 | 0.2407 |
| hide reserves + memory | 0.0000 | 0.0010 | 0.2461 |

Against eps-optimal, all three agents are near-perfect; the blind agent pays
essentially none of its uniform floor on-policy. Against random, **all three**
pay ~**0.24–0.25** — that shared cost is **coverage** (the game at this budget),
not aliasing; the full agent pays it too.

**(5,5), eps-optimal (25% noise), mean regret [min–max] over 3 seeds**

| observation | 300K episodes | 600K episodes |
|---|---|---|
| full | 0.0004 [0.0002–0.0005] | 0.0001 [0.0–0.0002] |
| hide reserves | 0.0007 [0.0003–0.0011] | 0.0004 [0.0003–0.0005] |
| hide reserves + memory | 0.0004 [0.0002–0.0005] | 0.0001 [0.0–0.0002] |

The blind–full gap at 300K (up to 0.0011 vs 0.0005) **does not persist** at 600K
(0.0003–0.0005 vs 0.0–0.0002) — consistent with **coverage**, not aliasing.
Memory tracks full at both budgets (identical Q-table sizes per seed).

**(5,5), random opponent, 600K episodes, mean regret [min–max]**

| observation | mean regret [min–max] |
|---|---|
| full | 0.034 [0.0246–0.0434] |
| hide reserves | 0.040 [0.0269–0.0495] |
| hide reserves + memory | 0.034 [0.0246–0.0434] |

Random play raises regret for **every** agent (including full) into the 0.02–0.05
range at (5,5) — again shared coverage, not a blind-specific penalty. Memory
matches full exactly (same audit seeds).

**Conclusion — and a correction to an earlier version of it.** The aliasing floor
is **provable yet behaviorally invisible**: the mask-blind uniform floor grows
**0.0805 → 0.1677**, yet the trained agents' realized on-policy regret sits far
below it. An earlier version credited that whole gap to *steering* (policies
avoiding aliased states). That over-simplified, and the repo's own numbers show
why — the gap decomposes into **two** effects, in different proportions by size:

- **Interface (the mask), not steering, does most of the work at (4,4).** Our
  agents never had the mask-blind interface: `QAgent` argmaxes over
  `get_legal_moves`, so its true floor is the **mask-aware** 0.0026, not 0.0805
  (Finding 4's interface table; the ladder in the preamble). At (4,4) the
  mask-blind floor *over the states the agent actually visits* is still **0.0766**
  (≈ the 0.0805 uniform floor) — so steering barely helps; the 30× drop is the
  mask. Tellingly, `excess_over_floor` (realized minus mask-blind floor on visited
  states) is **negative** (e.g. −0.019 to −0.026 at (5,5)): a genuinely mask-blind
  policy cannot pay *below* its own floor on its own trajectory — proof the mask,
  not steering, is paying the bill.
- **Steering dominates at (5,5).** There the mask-blind floor over visited states
  collapses to ~**0.017** (from 0.1677) — a ~10× cut from steering alone — and the
  mask takes it the rest of the way. So the decomposition is **size-dependent**;
  neither "all interface" nor "all steering" is right everywhere.

The honest one-liner: realized regret is a property of **interface *and*
trajectory**, not the mask-blind floor alone — and how much each contributes
depends on the size. Independent reimplementation matched **1,357,963** states and
**0.0805** at (4,4) to four decimals.

### 8. Is there a teachable "basic strategy"? Certified **yes** at (3,3) — certified refutations from (4,4) up
`experiments/depth_sweep.py`, `experiments/best_response.py`,
`experiments/kimi_gates.py`, `experiments/kimi_census.py`

**Thesis.** In Collapse3 a "strategy" in the everyday sense — a compact set of
rules you can teach — provably runs out: it works at the smallest board and is
exactly refuted beyond it, so competence becomes a *calculation*, not a rule.

This finding now has two halves: a **depth sweep** (how much lookahead shallow
policies need) and a **strategy-complexity crossover** (an externally proposed,
human-readable rulebook, certified exactly — the centerpiece). The crossover
table first, because it is the headline:

| size | rulebook v1, worst case | rulebook v2, worst case |
|---|---|---|
| (3,3) | forced loss (5–6 plies) | **DRAW — certified** |
| (4,4) | forced loss (5–6 plies) | forced loss (7–8 plies) |
| (5,5) | forced loss (5–6 plies) | forced loss (7–8 plies) |

Every cell is an **exact** result (both seats), not a sample: the best-response
solver enumerates the opponent's entire strategy space against the frozen
policy. Details below, after the depth sweep that frames them.

**The depth sweep** (reserves (4,4), seat P0, 200 games/cell). Most games have
a *strategy ladder*: a compact positional heuristic ("build lines, block
threats, hold the centre") gets a beginner to coherent play, and refinements
add strength. We test whether Collapse3 has one by grading **every** decision
against the oracle, separating the two variables a shallow policy can turn:
**evaluation quality** and **lookahead depth**. P0 at (4,4) is a **drawn**
seat, so any loss is the agent's own error. Regret is in **win/draw/loss units
(0–2)** — the same units as the aliasing floors (Finding 4), so the two are
directly comparable. The right baseline for a shallow policy is **random play**,
not 50%: critical states have 9–14 legal moves and only a fraction are optimal,
so guessing already scores ~0.36.

| policy | vs opt: draw/loss | opt-rate (critical) | WDL-regret (critical) |
|---|---|---|---|
| random baseline | 0.16 / 0.84 | 0.359 | 0.641 |
| myopic (own-alignment greed) | 0.01 / 0.99 | **0.000** | 1.000 |
| bead-count eval, 1-ply | 0.16 / 0.84 | 0.359 | 0.641 |
| bead-count eval, 2-ply | 0.76 / 0.24 | 0.851 | 0.149 |
| bead-count eval, 4-ply | 0.86 / 0.14 | 0.917 | 0.083 |
| bead-count eval, 6-ply | 1.00 / 0.00 | **1.000** | 0.000 |
| positional eval, 1-ply | 0.36 / 0.64 | 0.617 | 0.383 |
| positional eval, 2-ply | 0.99 / 0.01 | **0.997** | 0.003 |
| positional eval, 3-ply | 1.00 / 0.00 | 1.000 | 0.000 |

> **A one-line theorem behind the depth-1 collapse.** In Collapse3 *every* legal
> move changes the on-board bead differential by **exactly +1**: a placement adds
> one of your beads, a removal deletes one of the opponent's. So a one-ply
> bead-count evaluation is **constant across all moves** — it carries *zero*
> move-selection information, and the searcher is random-with-extra-steps. The
> byte-for-byte tie with the random baseline (0.359 critical, both) confirms it.
> This is why `0.359` is the score of one *provably degenerate* evaluation, **not**
> a generic one-ply number — the positional row (0.617) is the fair one-ply figure.

Three exact reads, one per variable:

- **The intuitive rule is the worst thing you can do.** "Maximise your own
  alignment" (`MyopicAgent`) is optimal on **0%** of critical decisions and loses
  **99%** of a drawn game — *below* random. In this game greed walks straight into
  the gravity/removal/Oops traps: following natural advice is *negatively*
  correlated with correct play.
- **Better evaluation, at fixed depth, helps but doesn't rescue.** Swapping the
  degenerate bead-count for a working positional eval lifts one-ply critical
  accuracy **0.359 → 0.617** — but it *still loses 64%* of a game it should draw.
  **No one-ply policy over a board-snapshot heuristic holds the draw**, however
  good the heuristic. (One ply over the *exact* values trivially does — but that
  is the deep search, not a snapshot rule; it is the very thing we claim you
  can't compress away.)
- **Lookahead is the decisive ingredient.** Holding the *same* evaluation fixed
  and adding plies: the positional eval jumps **0.617 → 0.997** with a single extra
  ply (2-ply) and is perfect at 3-ply; the blind bead-count eval climbs 0.359 →
  0.851 → 1.000, needing 6-ply. Either way competence is **bought in plies** — a
  good evaluation just needs fewer of them. WDL-unit critical regret collapses to
  ≈0 the moment real lookahead appears.

Two notes for the curve. The bead-count **2↔3 flatness** (0.851 → 0.848) is an
**odd/even parity effect** of fixed-depth minimax with a static leaf evaluation —
an extra ply that ends on the opponent's move can mislead the horizon score — not
seed noise. And against a **random opponent** every fixed-depth policy scores
*lower* on critical decisions than it does vs optimal (e.g. positional 2-ply drops
0.997 → 0.969), so **distribution-dependence survives even when lookahead is held
constant** — off-distribution states are harder at any fixed depth.

**The crossover: an externally proposed rulebook, certified and refuted.**
An earlier version of this finding closed with "any proposed basic strategy can
be graded exactly — we'd welcome candidates." Someone took us up on it (the
strategy was proposed by Kimi, an external model, in a multi-model review
round). We froze two formalizations of the proposal as named agents and built
the instrument to grade them:

- **The tool** (`experiments/best_response.py`). For *any* deterministic policy
  P, seat, and size, solve the one-sided game exactly: at P's turns the tree
  takes P's move (one branch); at opponent turns the opponent minimizes P's
  win/draw/loss outcome. Because the policy side never branches, the one-sided
  tree is tiny — a few hundred to a few thousand states even at (5,5) — so any
  frozen strategy gets a **certificate** (worst case ≥ draw against every
  possible opponent) or a **refutation** (a forced-loss line) in well under a
  second. It is a first-class, reusable instrument: submit a policy, get a
  proof.
- **The rulebook** (`KimiAgentV1`, frozen in `collapse3/agents.py`): take a
  win-in-1 if you have one (the only rule that removes); otherwise place on
  empty pegs — centre, corners, edges, in that order — then on opponent-topped
  stacks, then on your own; skip any move that hands the opponent a win-in-1,
  checking safety *within the current preference tier only*. Human-readable,
  five lines. It is exactly the kind of "basic strategy" most games have.
- **`KimiAgentV2`** (also frozen): identical, except the safety check escalates
  *across* tiers — scan all moves in preference order and take the first safe
  one anywhere. Not a repair of v1; a second formalization of the same
  proposal's ambiguous sentence, and the difference turns out to be the whole
  story at (3,3).

Reading the crossover table, in the order the claims should be believed:

1. **The positive result: a certified basic strategy exists at (3,3).** v2
   holds the draw from **both seats against every possible opponent** — an
   exact certificate, not a sample. At (3,3) the "no basic strategy" claim from
   earlier versions of this finding is **withdrawn**: a compact, teachable
   rulebook is a provably exact drawing policy there.
2. **The crossover: from (4,4) up, both formalizations are certified forced
   losses**, both seats, exploits 5–8 plies deep. "Calculable but not
   teachable" is restored *with boundaries*: **teachable at (3,3) is proven**
   (the certificate); at (4,4)+ what is proven is the refutation of *these*
   rulebooks — the general claim that *no* compact rulebook survives (4,4)+
   remains our conjecture, now with two certified refutations as evidence
   rather than none.
3. **The evaluation lesson — the finding's sharpest edge.** Before the
   best-response solve, v1 was evaluated the way agents usually are: play many
   games against a strong-but-noisy opponent. Against eps-optimal(5%) it went
   **1,199 of 1,200 games without losing** (1 loss; critical-decision
   optimal-move rate 0.993). Looks like a drawing policy. The exact solve then
   refuted it in **~0.03 seconds** with a **5-ply forced kill**. At
   eps-optimal(25%) the record is 9 losses in 1,200 — and we traced all nine:
   every one is the same bare-tower pattern as the certified exploit, i.e.
   noise occasionally stumbling onto the kill that adversarial search finds
   instantly. Distributional evaluation didn't just miss the flaw — it missed
   the *shallowest possible* refutation of the policy while grading it 99%
   optimal on critical decisions. (The originating session reported 0 losses in
   1,200 at "25% noise" under its own opponent implementation; the numbers
   above are this repo's, and the discrepancy is itself distribution
   sensitivity — see the measurement-problem preamble.)
4. **Exploit anatomy, and the tie to Finding 3.** The (3,3) seat-P1 kill:
   opponent stacks the *same corner peg* three times (`place 0, place 0,
   place 0`) while v1 plays centre and a corner — a bare tower v1 never blocks,
   because the block lives in tier T1 (opponent-topped stacks) and v1's
   tier-local safety filter never escalates past T0 (empty pegs) while empty
   pegs remain. Five plies, and the graded line shows the opponent playing
   **value-preserving moves throughout**: v1 defeats itself. The (4,4) seat-P0
   exploit is different in kind: it routes through a true-game opponent
   **blunder** — a value-losing move that v1 cannot punish, exactly Finding 3's
   inversion mechanism, now arising against a policy rather than a frozen plan.
5. **The patching regress.** v2 repairs precisely the hole the exploit walks
   through (safety escalates across tiers) and that repair *is* the (3,3)
   certificate — but at (4,4)+ it dies to a deeper trap: forks assembled just
   past its one-reply safety horizon, pushing the exploit from 5 to 7–8 plies.
   The census (`kimi_census.py`, all 477,960 decision states at (4,4)) shows v2
   halving v1's critical-state WDL regret (0.086 → 0.043) with its residual
   errors concentrated in the land-grab phase (beads 3–4) — where forks are
   prepared beyond one reply. Each patch buys depth by embedding a little more
   lookahead into the rules; the rulebook road leads back to search. We
   deliberately do **not** implement v3: the ladder is open, and the
   best-response solver will grade any rung anyone cares to submit.

**Scope, restated for the whole finding.** The depth sweep shows the tested
natural heuristics do not yield a strong one-ply policy while modest lookahead
does; the crossover shows a compact rulebook *can* be exactly right — at the
smallest size — and that certified refutation, not more games, is how its
limits were found. What none of this establishes is that no compact rulebook
exists at (4,4)+: refuting two candidates cannot rule out a third. The
falsifiability clause was exercised as designed — the claims in this section
were corrected twice in one review round, both times by an exact computation.

### 9. Throwing the game is provably hard
`experiments/sandbagging.py` (exhaustive, reserves (3,3)–(5,5))

Can a player force their *own* loss — deliberately let the opponent win? Two
exact solves, plus a weaker variant, answer it. Call the would-be thrower
"grandpa" and the opponent "grandson"; "grandson wins" means a line **or**
attrition win for the grandson's seat.

| reserves | forced throw?† | weak throw?‡ | P(grandson wins), random grandson |
|---|---|---|---|
| (3,3) | **no** | yes | 0.3133 first · 0.0648 second |
| (4,4) | **no** | yes | 0.5510 first · 0.2021 second |
| (5,5) | **no** | yes | 0.8001 first · 0.4062 second |

† *Forced throw* (binary minimax): grandpa plays to make the grandson win; the
grandson plays adversarially to **avoid** winning. NO in every cell and at both
seats — grandpa **cannot force his own defeat**. ‡ *Weak throw* (binary minimax):
grandpa plays only to guarantee he **never wins** (outcome ∈ {draw, grandson
win}) against a grandson trying to *make* him win. YES in every cell. The
probabilities are exact rationals (e.g. (3,3) first = 47/150, (5,5) first =
513274009/641520000) for a **uniform-random** grandson while grandpa throws
optimally — grandpa can *raise* the grandson's win chance sharply with size
(0.31 → 0.55 → 0.80) but never to 1.

**Why this needs enumeration (it isn't the trivial case).** In a pure *claim*
game — tic-tac-toe — forced-throw impossibility is a one-liner: the only way the
game ends is a player claiming their *own* line, so obviously you can't make the
opponent claim one for you. Collapse3 removes that free pass. Its **automatic**
win conditions — a removal's gravity cascade completing the *opponent's* line,
and the surviving-bead tiebreaker — give a thrower real routes to *hand over* a
win, so "can you force a loss?" is genuinely open a priori and had to be settled
by exhaustive solve. The finding is that even *with* those routes on the board,
an unwilling opponent can still refuse.

**Why a forced throw is impossible — the game's automatic win conditions cut
both ways.** *Attrition poisoning*: grandpa's only move besides placing his own
beads is **removing grandson beads** (you may only remove the opponent's), and
every such removal shrinks the grandson's material and tips the surviving-bead
tiebreaker *toward grandpa* — his one form of aggression feeds his own attrition
win. *Reverse-Oops*: placing risks completing grandpa's own line, and the
grandson can answer with removals whose gravity cascade completes a line among
**grandpa's** beads, handing grandpa the win by physics. So a grandson set on not
winning can always reach a draw or force a grandpa-win. The **weak** throw is
easier precisely because grandpa can simply decline to complete his own line and
let the game peter out — hence "can't force a loss, *can* force a non-win," the
finding's sharp edge.

**What a regret audit can and cannot see here.** Two precise statements, one
positive and one negative — an earlier version of this finding overclaimed the
positive one, and external review produced the exact counterexample that now
sharpens it.

- *Thrown value is visible (a theorem).* The game value moves against the mover
  only through the mover's own sub-value move. So an agent that actually
  *realizes* an outcome worse than the standing value — loses a drawn game on
  purpose — must play at least one **positive-regret** move, and on that move
  the oracle scores it exactly like a genuine blunder. Regret cannot certify
  *intent* (it cannot tell a liar from a novice), but surrendered value always
  shows up on the sheet.
- *Thrown intent is not (the counterexample).* Regret is **silent between
  value-equivalent moves** — and a strategic thrower exploits exactly that
  freedom. At (3,3) with the thrower moving first, *all nine openings are
  zero-regret* (each preserves the draw), yet against the random grandson they
  are not equivalent for throwing: an edge opening yields P(grandson wins)
  = 520/8019 ≈ 6.48%, the centre only 433/8019 ≈ 5.40%. The optimal thrower
  picks the edge and pays **zero** regret for it. Steering *within the optimal
  set* — toward positions where a fallible opponent is most likely to take the
  win — is invisible to any value oracle. The oracle audits **thrown value, not
  thrown intent**.

This is still a distinct use of the solver from Findings 1–8 — bounding how far
deliberate underperformance can go and what trace it must leave — but the sharp
version is a two-sided result: a thrower can *set up* a loss for free, yet
cannot *realize* one without logging regret, and (forced-throw) cannot guarantee
one at all against an unwilling opponent.

**Scope.** The forced-throw and weak-throw answers are theorems *at the
enumerated sizes* ((3,3)–(5,5)); misère at (14,14) remains to be run (the root
is now solved — Finding 11 — but full enumeration still caps at (5,5)). The
win-probability column is exact arithmetic but for one **opponent model** (a
uniform-random grandson), not a claim about any particular agent or about
children. Keep the two apart: "you cannot force a loss" is a theorem; "a random
opponent wins with probability 0.80 at (5,5)" is a model-dependent exact number.
And none of this transfers to real AI sandbagging as studied empirically — a
system aware it is being evaluated, with hidden capabilities and an open action
space, can trivially underperform (e.g. just answer wrongly); Collapse3 models
none of that. What it offers is a solved environment in which proposed
sandbagging *detectors* can be unit-tested against ground truth — and the
counterexample above is precisely such a test, one that a naive regret detector
fails.

### 10. Elo prefers the exploitable agent — the rating inversion, measured
`experiments/elo_tournament.py` (reserves (4,4), 9 agents, 3,600 oracle-graded games)

Elo is the field's default scoreboard, so we gave it the best possible setting:
a full round-robin among the repo's named agents (both seats, 100 games per
pairing), ratings fitted by maximum likelihood on the standard logistic model
(draws scored ½; order-independent, unlike online Elo updates; regularized by
two virtual draws against a fixed 1500 anchor so an unbeaten agent has a
finite rating). Every game was oracle-graded, so the same 3,600 games yield
three independent columns per agent: its **Elo** (pure performance), its
**mean WDL regret** over the tournament's own state distribution (competence),
and — for the deterministic policies — its **exact best-response worst case**
(Finding 8's certificate).

| agent | Elo | W/D/L | mean WDL regret | worst case (exact) |
|---|---|---|---|---|
| kimi-v2 | **1825** | 543/248/9 | 0.0090 | **forced loss** (7–8 plies) |
| optimal | 1804 | 505/295/**0** | **0.0000** | ≥ draw (theorem) |
| kimi-v1 | 1745 | 489/240/71 | 0.0338 | forced loss (5–6 plies) |
| pos-2ply | 1511 | 314/187/299 | 0.1453 | n/a (stochastic) |
| myopic | 1459 | 357/6/437 | 0.2187 | n/a (stochastic) |
| pos-1ply | 1439 | 330/24/446 | 0.2197 | n/a (stochastic) |
| one-ply-heur | 1432 | 328/15/457 | 0.2260 | n/a (stochastic) |
| bead-1ply | 1217 | 138/50/612 | 0.4073 | n/a (stochastic) |
| random | 1043 | 33/61/706 | 0.4830 | n/a (stochastic) |

**The headline is an inversion, not just compression.** External reviewers
predicted Elo would merely *saturate* at the top of a drawish game. Measured,
it does worse: **Elo ranks a certifiably exploitable rulebook above the
perfect player.** kimi-v2 — a certified forced loss at (4,4), the very size
the tournament was played at — out-rates the optimal agent 1825 to 1804
(**+21.1 Elo**). A nonparametric bootstrap over the recorded head-to-head
outcome multinomials (1,000 resamples, Bradley-Terry refit each time; no game
replay) puts a **95% CI of [5.1, 38.3]** on that gap, with **P(gap > 0) =
0.996** — the inversion is not a sampling fluke of a thin lead. The right-hand
columns tell the truth: optimal is the *only* agent with zero losses and zero
regret, and its worst case is a theorem where the table-topper's is a 7-ply
kill.

**The mechanism is structural.** The equal game is a draw, so "unbeatable"
caps what optimal play can earn; the surplus Elo comes from *converting* wins
against weak opponents. The optimal agent only preserves value — among equally
optimal moves it has no reason to prefer the one that punishes a blunder-prone
opponent, so it converts 79–90% wins against the bottom of the pool where the
rulebook converts 95–100%. Win-maximization against the fallible and
value-preservation against the perfect are **different objectives**, and Elo
measures only the first. (This is the benign mirror of Finding 9's steering
result: choosing *among* value-equivalent moves changes realized outcomes
without changing competence — there the choice hid intent from the auditor,
here it earns rating.)

**Elo's worst single prediction in the data.** kimi-v1's 286-point gap over
myopic implies an expected score of ~84%; the recorded head-to-head is a pure
**seat split** — each side wins *every* game in which it moves first, 50/50.
A rating model that assumes a scalar strength ordering cannot represent
"whoever moves first wins," which this pairing exactly is (Finding 6's
first-mover confound, surfacing inside the rating system).

**Scope.** This is one tournament: one pool, one size, 50 games per ordered
pair, one rating model (MLE Bradley-Terry with the stated anchor). The
*numbers* are pool-dependent by construction — that is the point — but the
inversion mechanism is structural in any drawish domain rated by average
results against a mixed pool, and here it is measured rather than argued.
Ratings answer "who beats whom, on average, in this pool"; they do not answer
"who plays correctly" — and in a solved game the two visibly disagree. And the
exploiting agent need not itself be strong: Finding 14(iv) exhibits a policy
that is random except on six memorized decisions yet forces a >99%-vs-random
champion to lose — a high rating measures who performs well in the tested pool,
not who is hardest to exploit. Accessible version in [FAQ #10](FAQ.md); numbers
in [`results/elo_tournament_latest.json`](../results/elo_tournament_latest.json).

### 11. The full game fell — (14,14) is a first-player forced win, found by accident
`experiments/full_game_value.py`, `experiments/horizon.py` (reserves (14,14))

This finding began as its own refutation. The plan was a "solvability horizon"
experiment: the (14,14) root was believed intractable (exact *enumeration* tops
out near (5,5), and an external session's ad-hoc solver could not solve even
ply 3 of a real full-size game within 4M nodes), so we built a node-capped
exact solver to walk finished games backward from the end and find the first
ply where exact grading becomes possible. The reproduction gate then failed in
the only direction gates never fail: the repo's solver — alpha-beta with a
bound-flagged transposition table, D4 symmetry folding, and win-first move
ordering — solved the **root** of the full game in **~96K nodes / 3.4
seconds**. There is no horizon. **Every ply of every game we played at
(14,14) is exactly graded.**

**The result.** The full 14-bead game is a **first-player forced true win**
(value +100), with a certified winning line 7 plies long. It is not close to
the edge of tractability, and it is not specific to 14:

| opening reserves | exact value |
|---|---|
| `(r, r)`, r = 1–5 | draw *(published table, unchanged)* |
| `(r, r)`, r = 6–14 | **first-player true win** |
| `(r0 ≥ 6, r1 ≥ 3)`, all 108 cells | first-player true win |
| `(r0 ≥ 2, r1 ≤ 2)`, r1 < r0 | first-player attrition win |

The full 14×14 grid is in
[`results/full_game_value_latest.json`](../results/full_game_value_latest.json);
it extends the published 6×6 table without changing a single cell, and the
mover-never-loses pattern holds everywhere.

**Why tractability was misjudged.** The state *count* explodes with reserves,
but alpha-beta only visits what the proof of the root value needs, and that
proof is small because **the forced win is shallow**. The mechanism is a rules
fact: legal moves do not depend on the reserve counts until someone's reserve
empties (placement legality tests `reserves > 0`, nothing else) — so the game
trees of (6,6) and (14,14) are *identical* until a player has spent six beads,
and the (6,6) win arrives well before that in every line the defense can
force. Extra material adds nothing the defender can use. Tractability tracks
the **depth of the forced win**, not the size of the state space — the mirror
image of chess endgames, which go exact when material *leaves* the board.
Collapse3 goes exact because wins arrive early.

**Verification.** A standing "intractable" claim died here, so the result ran
the full hardening battery before being written down:

1. **Independent clean-room engine.** The (14,14) root re-solved over
   `collapse3/reference_engine.py` (written from `rules.md` only) with a
   separate plain-TT alpha-beta sharing zero code with the main solver:
   **+100** (126K nodes).
2. **No transposition table, no symmetry folding.** A raw alpha-beta on the
   main rules engine — the two riskiest solver components removed entirely:
   **+100** (42.2M nodes, 16 minutes; recorded in the results JSON).
3. **Principal-variation replay.** Every position on a certified winning line
   re-verified with a fresh capped solve.
4. **Grid continuity.** All 36 previously published opening cells reproduced
   exactly; the pattern is uniform across 196 cells.

**The first exact full-game facts.** With the root solved, the horizon
experiment became a full-game grading battery
([`experiments/horizon.py`](../experiments/horizon.py); ten zoo pairings, five
seeds where stochastic, node cap 4M): **100% of moves graded in every game**,
across every pairing. First exact facts from real full-size games:

- **kimi-v2 self-play (deterministic):** 14 plies, P1 wins by line — and the
  grading shows P0 (playing the *winning* side of a solved game) **threw a
  forced win on three consecutive decisions** (plies 6, 7, 8, each WDL regret
  2). The externally proposed rulebook doesn't just lose at (4,4); at full
  size it stands in a won game and hands it over, three moves running.
- **pos-2ply and pos-1ply self-play:** first player wins in 7 (resp. 5) plies
  with **zero** graded blunders on either side — weak agents ride the
  first-player win without resistance the oracle can fault.
- **random-vs-random:** 6–17 plies, 0–8 blunders per game; the only pairing
  where the second player sometimes wins.

Along every graded trajectory a **consistency invariant** is asserted: the
exact value may change between consecutive plies only across a move graded
with positive regret (the mover-oriented value can never improve across the
mover's own move). This invariant is what exposed an earlier prototype that
stored alpha-beta cutoffs without bound flags — the same bug this project's
first review found in the original solver — and it now runs on every
trajectory, forever (`tests/test_horizon.py`).

**What this unlocks.** Optimal play at (14,14) costs a capped solve per move
(~seconds). The Elo tournament (Finding 10), the sandbagging solves (Finding
9), and the frozen-plan experiment (Finding 3) can all be rerun **at full
size with exact ground truth** — no approximate oracle needed. That was the
"future work" item; it is now just work.

**Scope.** Solving the root is not enumeration. Value-certified: the root
(and any position a capped solve reaches). Still capped at (5,5): everything
that requires visiting *all* reachable states — the aliasing floors, the
distribution-free census, the misère/sandbagging solves. And "the game is a
first-player win" is a statement about perfect play, not about play: as the
kimi trace shows, real agents stand in the won game and drop it. The session
that motivated this experiment reported a solvability horizon at ply 4 under
the same 4M-node cap; its trajectory, tail values, and grading all reproduce
exactly here (the gate passed), and the horizon difference is search strength
— symmetry folding and move ordering — not engine semantics.

### 12. A sibling game shows the floor's *shape* — bounded growth, and a hidden feature whose cost rises then falls
`experiments/threepeg_floor.py` (Three-Peg Collapse, reserves (2,2)–(14,14))

> **Scope — read this first. This is a *sibling* game, not a miniature of
> Collapse3.** *Three-Peg Collapse* restricts placements to the single row of
> pegs (0, 1, 2) and changes nothing else — stacking, gravity, the
> Collapse/removal and all five of its legality conditions, cooldown, the Oops
> rule, line and attrition wins, and the repo rule (no legal action → immediate
> attrition end) are byte-for-byte the same engine (it is a move-legality
> restriction, not a fork). But the geometry is different: only **8 of the full
> board's 49 winning lines** can ever fire (3 verticals, the row at 3 levels, 2
> staircases), which makes the sibling **attrition-dominant** — P0 wins by bead
> count at nearly every size, whereas real Collapse3 at (6,6)+ is a 7-ply forced
> **line** win ([Finding 11](#11-the-full-game-fell--1414-is-a-first-player-forced-win-found-by-accident)).
> Its findings are **parallel evidence about the same mechanics under different
> geometry**. They do **not** transfer to Collapse3 and must never be quoted as
> facts about it. The value is exactly the contrast: two environments, one
> engine, different boards — which lets us ask *which findings are caused by the
> mechanics and which by the geometry.*

**Why the sibling exists.** Finding 4's scaling claim — that a missing feature's
cost grows with the game — rests on the only two enumerable full-board points,
hide_reserves **0.0805** at (4,4) and **0.1677** at (5,5); the full board stops
being enumerable at (6,6). Two points show a *trend* but not a *shape*. Because
Three-Peg Collapse lives on one row, it is fully enumerable at **every** reserve
size — including **(14,14)**, the full game's reserve count, which has only
**94,824** reachable states and enumerates in ~2s — turning two points into a
complete 13-point curve (uniform-over-decision regret, WDL units, the same
machinery as Finding 4).

> **What "(14,14)" means here — reserve, not board capacity.** The single row
> holds at most **9 beads at once** (3 pegs × 3 levels), while **(14,14)** is the
> *reserve* each player starts with. These differ, and the mismatch is the
> point: 14 reserves far exceed a 9-cell board. Reserves above 9 are still real
> and consumable because the Collapse **destroys** a bead and frees its cell, so
> a game cycles place → removal → place and spends far more than 9 placements
> total; each distinct reserve count is a distinct state, so the reachable set
> keeps growing ((13,13) 85,243 → (14,14) 94,824). "The full game's reserve
> count" is chosen deliberately: we can enumerate the sibling at the same reserve
> size the 27-cell board never reaches — it is **not** a claim that the sibling's
> (14,14) is equivalent to Collapse3's (see the scope box).

| reserves | states | decisions | hide cooldown | Δcd | hide reserves | Δres |
|---|---|---|---|---|---|---|
| (2,2) | 154 | 55 | 0.0000 | — | 0.0000 | — |
| (3,3) | 1,108 | 616 | 0.0049 | +0.0049 | 0.0097 | +0.0097 |
| (4,4) | 4,352 | 2,869 | 0.0199 | +0.0150 | 0.0641 | +0.0544 |
| (5,5) | 10,505 | 7,435 | 0.0371 | +0.0173 | 0.1235 | +0.0593 |
| (6,6) | 18,624 | 13,614 | 0.0422 | +0.0051 | 0.1643 | +0.0408 |
| (7,7) | 27,773 | 20,674 | **0.0424** | +0.0001 | 0.1905 | +0.0262 |
| (8,8) | 37,338 | 28,080 | 0.0401 | −0.0023 | 0.2090 | +0.0184 |
| (9,9) | 46,919 | 35,562 | 0.0390 | −0.0011 | 0.2209 | +0.0119 |
| (10,10) | 56,500 | 43,044 | 0.0383 | −0.0006 | 0.2286 | +0.0078 |
| (11,11) | 66,081 | 50,526 | 0.0379 | −0.0005 | 0.2341 | +0.0055 |
| (12,12) | 75,662 | 58,008 | 0.0375 | −0.0003 | 0.2382 | +0.0041 |
| (13,13) | 85,243 | 65,490 | 0.0373 | −0.0003 | 0.2413 | +0.0031 |
| (14,14) | 94,824 | 72,972 | 0.0371 | −0.0002 | 0.2438 | +0.0025 |

The Δ columns are the point. Three exact reads, then the phase boundary that
unifies them.

**(i) The reserves floor grows steeply, then *flattens* — bounded over the
complete game-range, not unbounded.** hide_reserves climbs to a peak *increment*
of +0.0593 at (5,5), after which the increments collapse monotonically — +0.041,
+0.026, +0.018, …, +0.0025 — and the complete 2..14 curve **flattens sharply near
0.25** (exact value at (14,14): **0.2438**). That is finite-range flattening,
not a proved mathematical asymptote beyond 14 — but 14 is the real game's material
limit, so the complete game-range curve is what matters. So in the sibling the
cost of the missing reserve feature is **bounded**. This does **not** retract
Finding 4: for the sizes actually measured on the full board the growth claim
stands exactly as written. What the curve adds is *shape* — and it is a caution
about extrapolation, because the full board's only two enumerable points,
(4,4) and (5,5), sit squarely on the **steep early section** of a flattening
curve. Two points on that section could suggest unbounded growth or a plateau
equally well; nothing in them reveals which. In the sibling it flattens; on the
full board the shape beyond (5,5) **remains unknown** (see the cross-reference
now in Finding 4). The flattening is visible even in the raw state count: from
**(8,8) onward each added reserve contributes a constant +9,581-state shell**
(37,338 → 46,919 → 56,500 → … each exactly +9,581), and the observation-group
count **freezes at 1,441** — once the reserve exceeds what a 9-cell board can be
*about*, the marginal bead adds another identical layer of positions and no new
structure, which is precisely why the floors flatten.

**(ii) The cooldown floor is *non-monotonic* — it rises, peaks at (7,7), then
falls.** hide_cooldown climbs to **0.0424** at (7,7) and then *declines* and
plateaus near ~0.037. A hidden feature's cost that goes **up and then down** with
game size is the result **no two-point extrapolation could ever have produced** —
from (4,4)→(5,5) (0.0199 → 0.0371) one would only ever predict continued growth.

**(iii) One phase boundary near (6,6)–(7,7), three signatures.** The exact
openings tell the same story from the strategy side. The empty-board root is a P0
attrition win at every size ≥ (3,3), but the *value of playing the centre peg*
inverts across a single boundary (peg 0 and peg 2 are symmetric ends throughout):

| reserves | centre (peg 1) | ends (pegs 0, 2) | reading |
|---|---|---|---|
| (3,3)–(4,4) | P0 attrition | P0 attrition | everything wins |
| (5,5) | **P0 attrition** | draw | centre *uniquely* wins |
| (6,6) | draw | P0 attrition | centre only draws |
| (7,7)–(14,14) | **P1 line win** | P0 attrition | centre **loses to a line**, then frozen |

Three signatures land on the same transition near (6,6)–(7,7): the centre-strategy
inversion **freezes** (identical from (7,7) up), the cooldown floor **peaks**, and
the reserve-floor increments **collapse**. The unifying reading — labelled as a
**hypothesis**, not a theorem — is that the 9-cell board (here, 3 pegs × 3 levels)
becomes the binding constraint. Below the boundary material is scarce and the
reserve *count* discriminates between positions; above it the board fills before
reserves matter, so extra beads stop carrying information and every added-material
signal flattens at once. Finding 13 tests the directional prediction of that
hypothesis on a larger sibling.

**(iv) Hypothesis (labelled as such — not a result).** *If* board capacity sets
the boundary, then a larger board should push the boundary later: a 27-cell board
would saturate far beyond a 9-cell one, so the real 14-bead game may **never leave
its pre-saturation regime**. This is a conjecture the present data cannot settle.
The named test is a **6-peg (two-row, 18-cell) sibling** as the middle data point
between 9 and 27 cells. **That test is now run — see
[Finding 13](#13-a-second-sibling-six-pegs-shows-the-boundary-moves-with-capacity--finding-12iv-tested):**
every boundary signature is delayed by ~2 reserve sizes on the 18-cell board,
confirming the *direction* (capacity, not mechanics, sets the boundary), though
the un-folded state space explodes (9.15M states at (7,7)) and six-peg's own
boundary stays out of enumeration reach.

**Exactness re-verified for the variant (A3).** Finding 4's Exactness lemma (the
charity rule never fires for the single-hide ablations, so those floors are exact
optima rather than deflated bounds) was proven by enumeration on the *full* board;
it is not assumed to carry. The scan was re-run here at every reserve size 2..14
and `no_common_legal_action == 0` for both hide_cooldown and hide_reserves
throughout — so every floor in the table above is an **exact optimum**, not a
bound. (`hide_both` is the deliberate contrast: hiding both fields gates legality,
the charity rule fires, and the column is a deflated lower bound, exactly as in
Finding 4.) All numbers, the openings gate, and the lemma scan are guarded by
`tests/test_threepeg_floor.py` and recorded in
[`results/threepeg_floor_latest.json`](../results/threepeg_floor_latest.json).

### 13. A second sibling (six pegs) shows the boundary *moves* with capacity — Finding 12(iv), tested
`experiments/sixpeg_floor.py` (Six-Peg Collapse, reserves (2,2)–(7,7))

> **Scope — same discipline as Finding 12. This is a *third geometry*, not a
> miniature of anything.** *Six-Peg Collapse* restricts placements to the top two
> rows (pegs 0–5) and changes nothing else — byte-for-byte the same engine, a
> move-legality restriction only. The board is now **18 cells** (6 pegs × 3
> levels) and **16 of the full board's 49 winning lines** can fire (6 verticals,
> 2 rows at 3 levels, 4 staircases) — double the three-peg sibling's 8, so it is
> less attrition-dominant. Its numbers are **parallel evidence about the same
> mechanics under a third board**; they do **not** transfer to Collapse3 and must
> never be quoted as facts about it. The value is the *three-point contrast*: one
> engine, boards of 9, 18, and 27 cells.

**Why the second sibling exists.** Finding 12(iv) made a falsifiable conjecture:
*if board capacity (not the mechanics) sets the phase boundary near (6,6)–(7,7)
in the three-peg sibling, a larger board should push that boundary later.* Six
pegs are the named middle point between the 9-cell single row and the 27-cell
full board. This finding runs that test.

**The feasibility wall (a result in itself).** Unlike the single row, the two-row
board is *not* capped small: the un-folded reachable set explodes ~3–4× per
reserve, reaching **9,152,201 states at (7,7)** — already ~100× the three-peg
sibling's *entire* (14,14). Enumeration is memory-bound around (7,7)–(8,8), and
the sibling's own predicted boundary (~(12,12)+) is **out of reach by brute
force**. So this finding cannot show six-peg's boundary *peak*; it shows the
boundary's *location has moved*, which is exactly what 12(iv) predicted.

**All three signatures are delayed by ~2 reserve sizes on the 18-cell board.**
The cooldown floor is the sharpest test, because on three pegs it *peaks at
(7,7)* and then falls:

| floor | three-peg (9 cells) | six-peg (18 cells) |
|---|---|---|
| hide_cooldown (5,5) | 0.0371 | 0.0112 |
| hide_cooldown (6,6) | 0.0422 | 0.0142 |
| hide_cooldown (7,7) | **0.0424 — peak, then declines** | **0.0167 — still rising (+0.0025)** |

At (7,7) three-peg has turned over; six-peg is still on the rising limb, at a
much lower level. The reserve floor tells the same story — its increments, which
on three pegs **collapse** after the (5,5) peak, roll off far more gently on six
pegs (increment still **+0.0529** at (7,7), vs three-peg's +0.0262):

| hide_reserves (increment) | three-peg | six-peg |
|---|---|---|
| (5,5) | 0.1235 (+0.0593) | 0.1240 (+0.0622) |
| (6,6) | 0.1643 (+0.0408 ← collapsing) | 0.1790 (+0.0550) |
| (7,7) | 0.1905 (+0.0262) | 0.2319 (**+0.0529 ← barely rolling off**) |

And the exact openings — the strategy side — are shifted by the same ~2 sizes.
The empty-board root becomes a P0 attrition win at **(3,3)** on three pegs but
only at **(6,6)** on six pegs; the "centre peg uniquely wins" phase lands at
**(5,5)** on three pegs and at **(7,7)** on six pegs; and the centre's collapse
into an outright *line loss* — reached by (7,7) on three pegs — has **not
appeared at all** in the feasible six-peg range:

| phase | three-peg | six-peg |
|---|---|---|
| root becomes a P0 win | (3,3) | (6,6) |
| centre uniquely wins | (5,5) | (7,7) |
| centre draws | (6,6) | not reached |
| centre loses to a line | (7,7)+ | not reached |

**Reading.** Doubling the board from 9 to 18 cells delays every boundary
signature by ~2 reserve sizes and does **not** relocate it to a *different kind*
of transition — same three signatures, later. That is direct evidence that the
boundary is set by **board capacity, not the mechanics**: the mechanics are
identical across all three siblings; only the geometry changed, and the boundary
moved with it. It also sharpens Finding 12(iv)'s corollary: if 18 cells already
pushes the boundary out toward ~(12,12), the 27-cell real game plausibly spends
its entire 14-bead reserve budget in the **pre-saturation, still-rising** regime
— a statement offered as parallel evidence, **not** a fact about Collapse3 (which
this experiment cannot enumerate). What remains unmeasured is the exact scaling
*law*: the direction is confirmed on all three signatures; the magnitude (and
six-peg's own peak) is beyond enumeration.

**Exactness re-verified for the second variant.** The single-hide charity rule
never fired at any measured size (`no_common_legal_action == 0` for both
hide_cooldown and hide_reserves through (7,7)), so every six-peg floor above is
an **exact optimum**, not a deflated bound — the lemma holds under a third
geometry. (`hide_both` again gates legality and its charity fires, the deliberate
contrast.) All numbers, the census, the opening gate, and the lemma scan are
guarded by `tests/test_sixpeg_floor.py` and recorded in
[`results/sixpeg_floor_latest.json`](../results/sixpeg_floor_latest.json).

### 14. Self-play saturates: flawless on its own trajectories, weaker off them, and it trips over the phase boundary
`experiments/threepeg_selfplay.py` (Three-Peg Collapse, reserves (4,4)–(7,7), 5 seeds)

> **Scope — sibling, parallel evidence.** Same discipline as Findings 12–13: this
> is the Three-Peg *sibling*, not Collapse3, and its numbers never transfer. What
> is new is the *agent*: this is the first **learned-agent** result on the sibling,
> and the first **true self-play** anywhere in the repo. Everything else trains
> *against the oracle* ([Finding 2](#2-performance-and-competence-come-apart),
> [Finding 7](#7-the-floor-is-real-but-realized-regret-slips-beneath-it--interface-first-then-steering)); here **one shared Q-table drives both seats with zero
> oracle access during training** (Monte-Carlo, terminal reward only). The oracle
> is used *only afterwards*, to grade the frozen policy exactly. Un-folded
> throughout (the sibling is not grid-symmetric). Seeds are fixed, so every number
> is deterministic and reproducible.

The sibling is exactly solvable at *every* reserve size, so a self-play agent can
be graded across the whole (6,6)–(7,7) phase boundary — which the full board
(enumerable only to (5,5)) never permits. Two observation regimes: `full` (no
aliasing) and `hide_reserves` (the load-bearing lossy feature).

**(i) Near-perfect where it plays; globally competent but *eroding* — the
baselined version.** With **full** observation there is no aliasing floor, so
uniform (all-decisions) regret is a coverage/learning gap. A global regret of
"0.215" means nothing without a scale bar, so every row prints one: a
**uniformly-random legal policy** graded the same way (exact, closed-form, no
sampling — the mean legal-move regret per state, averaged over the same
denominator).

| (r,r) | states visited | regret on its own games | regret over ALL states | random-policy baseline | seat-0 forced below a win |
|---|---|---|---|---|---|
| (4,4) | 72% | 0.026 | 0.064 | 0.198 | 0 / 5 |
| (5,5) | 68% | 0.038 | 0.098 | 0.287 | 0 / 5 |
| (6,6) | 57% | 0.044 | 0.149 | 0.351 | 1 / 5 |
| (7,7) | 42% | **0.045** | **0.215** | **0.394** | **2 / 5** |

So the agent is **not** "blind everywhere" — globally it removes about two-thirds
of a random policy's regret at (4,4). The honest failure is subtler and shows up
only *because* the baseline is there: as the game grows the agent stays
near-perfect on its own trajectories (~0.045), while its **global edge erodes** —
from removing **67%** of the random baseline's regret at (4,4) to just **45%** at
(7,7) — and its coverage falls to 42%. On-policy regret alone would report a
flawless agent that is not getting worse; the baselined global number reports a
policy whose competence *off its own lines* is decaying. That disagreement — one
policy, two distributions — is the project's core thesis, reproduced by self-play
with no oracle in the loop. (Caveat, stated rather than hidden: this is a fixed
50k-episode budget, so part of the raw growth is coverage decay, not pure skill.
That is exactly why the random baseline and the coverage column sit *beside* the
regret, never blended into a single "competence" scalar — the denominator
discipline of Finding 5, applied to a learned agent.)

**(ii) The "perfect-looking" policy is exactly exploitable — increasingly so with
size.** Freezing each policy and letting the worst-case adversary reply
(`experiments/best_response.py`, exact, un-folded) certifies the outcome. The
empty-board root is a P0 forced win at every size, so **seat 1 is theoretically
lost regardless** — its best-response loss is the game value, *not* an exploit,
and is reported only for completeness. The informative seat is the winnable
seat 0: at (4,4)/(5,5) self-play secures the win in all 5 seeds, but at (6,6) one
seed and at (7,7) **two of five** can be forced off their won game to a draw or
loss. A policy that goes flat/undefeated in self-play is provably not robust.

**(iii) The headline: a self-play agent trips over the exact phase boundary.** A
reserve-blind (`hide_reserves`) policy has a *size-independent* opening — the
empty-board observation is identical at every reserve. Trained at **(5,5)**, where
[Finding 12](#12-a-sibling-game-shows-the-floors-shape--bounded-growth-and-a-hidden-feature-whose-cost-rises-then-falls)
proves the centre *uniquely* wins, **all 5 seeds learn to open on the centre** —
and carry it straight across the boundary:

| opening chosen at (5,5) | value at (5,5) | at (6,6) | at (7,7) |
|---|---|---|---|
| centre (peg 1), all 5 seeds | **win** | draw | **loss** |

At (7,7) the centre inverts to a forced P1 line loss
([Finding 13](#13-a-second-sibling-six-pegs-shows-the-boundary-moves-with-capacity--finding-12iv-tested)),
and best-response confirms **5/5 seeds are forced to lose**. The agent learned a
rule that was *provably optimal in its training regime* and applied it after the
environment crossed into a different strategic regime — a clean, exact
distribution-shift failure landing on the enumerated boundary.

**Caution (do not over-read the representation comparison).** Under a fixed
budget the `hide_reserves` agent posts *lower* global regret than the full-state
agent at the larger sizes (~0.06 vs 0.215 at (7,7)) — but that is a coverage
artifact (hiding reserves collapses many states to one key, so the smaller table
is better covered), **not** evidence that hiding information improves competence:
the very same reserve-blind agent is forced below a win in **5/5** seeds at (5,5),
exactly where reserves are decisive. Low average, fatal worst case — the
aggregation lesson, now at the level of *representation*. (The same
average-versus-worst-case failure has been observed independently via a second
learning route — supervised oracle imitation — as external parallel evidence; it
is not reproduced in-repo here.)

All numbers are guarded by `tests/test_threepeg_selfplay.py` and recorded in
[`results/threepeg_selfplay_latest.json`](../results/threepeg_selfplay_latest.json).

**(iv) The exploiter need not itself be strong.** Part (ii) certifies the
champion is force-losable; `experiments/weak_exploiter.py` characterizes the
*forcing policy*, and the result is sharper than "the champion is exploitable."
Take the reserve-blind champion trained at (5,5) and transferred (its opening is
size-independent). It beats a uniformly-random opponent **99.5%** at (6,6) and
**99.1%** at (7,7) — yet a policy that plays **uniformly at random everywhere
except six memorized decisions** forces it to lose (12-ply line). Those six
decisions are **0.044%** of the reachable decision states at (6,6) (0.029% at
(7,7)); the shipped-style full-state champions (trained natively, >99.99% vs
random) fall the same way to **seven** memorized decisions.

| champion | vs-random win | force-losable? | exploiter differs from random on |
|---|---|---|---|
| transfer→(5,5) | 99.3% | **no** (draw only) | — |
| transfer→(6,6) | 99.5% | yes (12 plies) | **6 states — 0.044%** |
| transfer→(7,7) | 99.1% | yes (12 plies) | 6 states — 0.029% |
| local (6,6), seed 2 | 99.99% | yes (14 plies) | 7 states |
| local (7,7), seed 0 | 99.997% | yes (14 plies) | 7 states |

So "the strong-looking agent is exploitable" becomes "…exploitable by something
almost completely incompetent at the game." Two caveats are kept in the open
because they bound the claim:

- **The "indistinguishable from random" measure is nearly definitional.** A
  policy that is random except on *k* states cannot differ from random's uniform
  regret by more than *k* / (#decision states) — so its 99.9% ratio is a
  restatement of "differs on six states," not independent evidence. The
  non-trivial content is that those six decisions **suffice**, which is why the
  *mapped-fraction* is the number we lead with.
- **This refutes one frozen, deterministic victim — it is not a strategy.** A
  deterministic champion has exactly one reply per turn, so a single memorized
  *line* forces the loss. Against a stochastic or adaptive champion a forcing
  *tree* (many more decisions) would be needed. This is a hard-coded key to one
  lock, not general skill.

The real-system analogue is Wang et al. (2023), *Adversarial Policies Beat
Superhuman Go AIs*: a weak, specialized adversary — itself beatable by human
amateurs — defeats a superhuman KataGo network. Same structure (strong
generalist victim, weak specialized exploiter); the honest difference is that
their adversary used **search / a forcing tree** against a searching victim,
where the frozen-champion case here is the degenerate single-line version, and
that Collapse3's exploiter is **exact and certified** rather than another
trained net. We do not claim the failure mode *transfers* from a board game to
Go; we point at a documented case where a structurally identical failure
already occurred. Numbers guarded by `tests/test_weak_exploiter.py`, recorded in
[`results/weak_exploiter_latest.json`](../results/weak_exploiter_latest.json).

### 15. Knowing when to look — the exact value of information is sparse
`experiments/info_policy.py` (reserves (3,3), (4,4))

Finding 4 prices a *fixed* missing feature: what does hiding reserves cost a
memoryless agent that never gets them? The adaptive question is more like the one
a real agent faces — *when should it decide its current view is insufficient and
go acquire more before acting?* Because the game is solved, the answer is exact.

For each coarse-observation group we compare two options: **act now** on the
coarse view (pay that group's aliasing regret), or **query** the finer view
(split the group into its finer sub-groups, each acting on its own best common
move). The per-group minimum is an exact **information policy** — an oracle
labelling of exactly where the coarse representation is genuinely insufficient.
It is the same enumerated, uniform-over-decision, opponent-independent machinery
as the aliasing floor, lifted from "one feature, always missing" to "one
feature, acquired on demand." We measure two axes, each isolating one
information good (the finer view strictly refines the coarser, so a query only
ever *splits* a group):

| axis | coarse view | queried good | (3,3) query rate | (4,4) query rate |
|---|---|---|---:|---:|
| **reserve** | board+turn+cooldown+**mask** | reserve counts | **0.000%** | **1.071%** |
| mask | board+turn+cooldown | legal-action mask | 0.827% | 13.212% |

The **reserve** axis is the load-bearing one — the mask is normally free, so
"querying" it is only illustrative, whereas reserves are a genuine hidden good.
Three things fall out:

- **Missing ≠ useful (at (3,3)).** Reserves are aliased away in 36% of decision
  states, yet on the mask-aware interface the exact `hide_reserves` floor is
  **0.0** — so the optimal number of reserve queries is **zero**. Objectively
  incomplete information can have no decision value.
- **An information transition ((3,3) → (4,4)).** From (4,4) up the mask-aware
  reserve floor becomes nonzero (0.0026278), and now there *are* states an
  optimal agent must inspect — but **only 1.071%** of them (5,120 / 477,960).
  That sub-1% of decisions carries the **entire** irreducible reserve floor:
  querying there drives the floor to exactly **0.0**. The value of information is
  real but **sparse**.
- **Sparse vs. diffuse goods.** The two axes look opposite. Reserves matter in a
  thin, decisive set (1.071%); the legal-mask is *diffusely* useful (13.212% at
  (4,4)) and querying it everywhere only reduces the view to the mask-aware
  interface — its residual floor **is** the reserve floor. Not all missing
  information is missing in the same shape.

An exact cross-check ties this to Finding 4: the (4,4) reserve axis flags exactly
**1,256** beneficial-query observation groups — identical to the
`masked_conflict_groups` count in `masked_floor`. The states where reserves help
are precisely the observation groups that contain an optimal-action conflict.

Attach a per-query cost and the policy sharpens into three measurable regimes:
**under-observation** (act coarse, eat the floor), **over-observation** (query
everywhere, pay needless sensing cost), and **selective competence** (query only
the critical states). On the (4,4) reserve axis the objective (residual regret +
cost × query rate) keeps querying the sparse set until the per-query cost exceeds
~0.25 WDL units, past which it is cheaper to accept the 0.0026278 floor.

> **Framing / prior art.** This is an exact, game-based instance of the *value of
> information* (Howard 1966) and of metareasoning about when to sense or compute
> (Russell & Wefald 1991) — the contribution is exactness on a fully solved game,
> not the concept. The mask axis is a semi-artificial "information good" (the
> mask ships for free), included only to contrast sparse against diffuse VOI; the
> honest headline is the reserve axis. The heavier (5,5) census (12.7M states) is
> left as a follow-up; the machinery already accepts it.

Numbers guarded by `tests/test_info_policy.py` (the (3,3) census is recomputed
live to the digit; (4,4) is guarded from the recorded file), recorded in
[`results/info_policy_latest.json`](../results/info_policy_latest.json).

### 16. Generalization is not robustness — a trained net that generalizes is still certifiably force-losable
`probes/` (torch-dependent exhibit; reserves (4,4))

Findings 3, 10, and 14 show that *bad or average* metrics — win rate, Elo,
on-policy regret — flatter a weak agent. This finding pushes the same knife into
the **best** average-case metric there is: a distribution-controlled held-out
accuracy with a near-zero train/test gap. A small MLP trained on exact solver
labels for a fraction of the 477,960 (4,4) decision states plays **~98%**
optimally on held-out states it never saw (0.933 optimal even at a 0.5%,
leakage-light train fraction; it extrapolates to a 1,200-state (5,5) sample at
0.923). By every generalization test, the network learned the game.

It is nonetheless a **certified forced loss**. Frozen and handed to the repo's
exact `best_response` solver from the drawn (4,4) root, **all six** trained
nets — two independent architectures (WDL classifier, raw-value regressor) ×
three seeds — are forced to lose from **both** seats (**12 of 12**
certifications), in this reproduction at depth 6 (seat 0) / depth 5 (seat 1). The adversary wins by playing deliberate
true-game blunders that only work against the specific net (the organic
inversion of Finding 3), and the first value-losing move it exploits lands on
held-out and unused states — pure generalization failures the audit could never
have flagged, because the failure is not in the distribution.

> **Why the two gaps diverge.** Average-case competence means the error set is
> *small*; worst-case robustness in a drawn game means it is *empty or
> unreachable*. A 2–5% critical-error rate over ~478K states is thousands of
> candidate errors, and the best-response solver needs only one to be
> force-reachable. "It generalizes" and "it is adversarially safe" are different
> claims, and the gap between them is not estimable from any average-case number.
> Unlike *empirical* adversarial-ML robustness (attack-relative, broken by the
> next attack) — and unlike *certified* robustness, which is attack-independent
> but local to a perturbation ball — this certificate is **global and
> game-theoretic**: the strongest adversary that exists, covering every reachable
> line, with a fully traceable exploit.

> **Scope.** Trained neural policies at (4,4), small MLPs on raw features — an
> exact *exhibit*, not evidence about frontier systems. Nothing here is a fact
> about Collapse3's solved values. The 12/12 forced-loss outcome is the
> reproducible claim; exact depths, lines, and neural decimals drift with torch
> build. Full writeup, tables, exploit anatomy, and the setup-dependent
> failure-anatomy reconciliation: **[`docs/NEURAL_EXHIBIT.md`](NEURAL_EXHIBIT.md)**.

Numbers bound to [`results/neural_best_response_latest.json`](../results/neural_best_response_latest.json)
and [`results/matrix.json`](../results/matrix.json) (the 12-certification record), and
guarded by `tests/test_neural_exhibit.py` (deterministic counts and the all-forced-loss
outcome; neural decimals are threshold-checked, not pinned).

### 17. Passing a test rules out less than you think — the strongest player is not the strongest tester
`experiments/evaluation_equivalence.py` (reserves (3,3)+(4,4), exact, pure-Python)

> *We don't evaluate agents. We evaluate the distribution of states a specific
> opponent induces. If the opponent isn't permitted to play badly, we're relying
> on it to wander into the danger zone by chance — and a perfect opponent cannot
> wander there at all. The states that would expose the weakness are exactly the
> ones optimal play refuses to enter.*

Every finding above grades a *given* agent. This one grades the **evaluation
itself**: after a candidate passes, how bad can a policy still be while remaining
consistent with everything the evaluation observed? An evaluation pins the
candidate to the reference at the states it inspected (set **C**); over *all*
policies compatible with those observations we compute the exact **compatible
outcome range** `(worst, best)` by best-response with pinned nodes — inside `C`
the mover is forced to the reference, outside `C` the adversary drives it. `best`
is always the game value (an assertion, not a finding); `worst` is the residual
the evaluation failed to rule out.

**The headline (transcript level, Gate C).** Pin the *complete* benchmark
support under two opponent families:

| size | seat | optimal-only opponents | all-legal opponents |
|---|---|---|---|
| (3,3) | 0 | 41 pinned → **forced loss** | 96 pinned → draw ✓ |
| (4,4) | 0 | **10** pinned → **forced loss** | 447 pinned → draw ✓ |
| (4,4) | 1 | 1,241 pinned → **forced loss** | 3,982 pinned → draw ✓ |

An evaluator that tests only against **optimal** opponents can inspect as few as
**10 decisions** and certify nothing — a policy that passes it perfectly is still
a *certified forced loss* — while the same protocol against **all-legal**
opponents rules the loss out. The effect is seat-0-only at (3,3) and **two-sided
at (4,4)**: it grows with the board. The subset direction is structural (fewer
constraints certify less); the *finding* is that restricting to optimal-opponent
play — the natural "test against strong bots" choice — is exactly what drops the
inspected set below the certification threshold, because **a perfect opponent
refuses to enter the objectively-losing lines where the candidate's specific
weakness lives**. Opponent strength and evaluator strength are distinct
properties. This is the same mechanism as [Finding 14](#14-self-play-saturates-flawless-on-its-own-trajectories-weaker-off-them-and-it-trips-over-the-phase-boundary)
and [Finding 16](#16-generalization-is-not-robustness--a-trained-net-that-generalizes-is-still-certifiably-force-losable),
where the adversary wins via true-game blunders — stated here as a property of
the evaluation protocol.

**H2 falsified (Gate D).** Strategic depth-limited coverage buys nothing over
depth-matched *random* coverage from the **same** exposure universe: whenever
strategic certifies, same-universe random certifies too (0/30 force-losable);
random from the whole state space never certifies (30/30). It is the universe,
not the selection. Reported as a plain falsification, not hidden.

**Gate B.** Finite games are inefficient: ~3,000 transcript-audited games are
needed to reach the draw certification that exact all-legal coverage nails with
96/538 decisions.

**How weak is the missed flaw? (Gate A).** The minimum-cardinality question:
how *few* non-canonical decisions must a policy that passed the optimal-opponent
evaluation still make to be force-losable? Exactly by lexicographic
shortest-path (forced-loss → fewest candidate mutations → least depth), at
**(3,3) seat 0 the answer is one** — a single non-canonical decision, first
reachable at depth 6, agreeing with the reference everywhere the perfect
opponent looked. The surviving flaw a strong-opponent test misses can be one
memorized mistake — the exact, minimum-cardinality cousin of the weak exploiter
(Finding 14).

This is a **transcript-level** result only — the `grade` and `outcome`
(win-rate/Elo) rungs are deliberately quarantined as unsound to fake by pinning
actions. Exact and pure-Python (no torch, no sampling in the headline). Full
tables, the observation-level hierarchy, and limitations:
**[`docs/EVALUATION_EQUIVALENCE.md`](EVALUATION_EQUIVALENCE.md)**. Numbers bound to
[`results/evaluation_equivalence_latest.json`](../results/evaluation_equivalence_latest.json)
and guarded by `tests/test_evaluation_equivalence.py` (reproduce-or-abort Gate 0
+ the (4,4) headline). This is **not a new method** — it is the exact,
game-theoretic instance of surviving-mutant reasoning and an identified set;
prior-art inventory and the honest novelty verdict ("new exact application, not
new method") in [`docs/EVALUATION_EQUIVALENCE_PRIOR_ART.md`](EVALUATION_EQUIVALENCE_PRIOR_ART.md).

### Where the difficulty lives
`experiments/structural_census.py` (reserves (4,4))

54% of reachable states are drawn; removal ("the Collapse") is the **only**
optimal move in ~78K states, so the removal mechanic is genuinely load-bearing.
Instant-loss ("Oops") removals exist but are rare (16 states).

---

## Relation to prior work

In one line: Collapse3 is an exact, game-based demonstration of the
memoryless-policy problem Littman (1994) formalized — lossy observations create
unavoidable aliasing costs, memory removes them, and on-policy performance can
still look fine depending on the state distribution. What Collapse3 adds is that
every part of that sentence becomes an *enumerated number* rather than a
worst-case bound.

The *phenomenon* behind Finding 4 — that a **memoryless** policy is provably
suboptimal when its observation aliases states that need different actions — is
classical, not new here:

- **Whitehead & Ballard (1991)**, *Learning to perceive and act by trial and
  error* — names the failure mode: **perceptual aliasing**.
- **Littman (1994)**, *Memoryless policies: theoretical limitations and practical
  results* — proves that finding a satisficing/optimal deterministic memoryless
  policy under aliasing is intractable, and bounds its performance below the
  **complete-information** agent (the omniscient lower bound) — the same
  construction as our floor. It also shows a single **bit of memory** can restore
  optimality — the antecedent of Finding 7.
- **Singh, Jaakkola & Jordan (1994)**, *Learning without state-estimation in
  POMDPs* — model-free learning for this policy class, including where stochastic
  memoryless policies help.

Finding 15 (deciding *when* to acquire a missing feature) sits in a third
classical line — the **value of information** and metareasoning about sensing/
computation:

- **Howard (1966)**, *Information Value Theory* — the expected value of acquiring
  a signal before deciding; Finding 15 computes it exactly per observation group.
- **Russell & Wefald (1991)**, *Principles of Metareasoning* — treating "gather
  more information / compute more vs. act now" as itself a decision. Collapse3
  adds an exact, enumerated instance on a solved game, where the query policy and
  its residual regret are certified rather than estimated.

A separate line of prior work is the real-system analogue of Finding 14(iv)
(exploitable-champion / weak-exploiter):

- **Wang et al. (2023)**, *Adversarial Policies Beat Superhuman Go AIs* (ICML) —
  a weak, specialized adversary (itself beatable by human amateurs) defeats a
  superhuman KataGo network. Same structure as Finding 14(iv) — strong
  generalist victim, weak specialized exploiter. Collapse3 adds an **exact,
  certified** version of that structure: the exploiter is a provable
  best-response, not another trained net. We do **not** claim the failure mode
  transfers from a board game to Go; we cite a documented case where a
  structurally identical failure already occurred (their adversary used a
  forcing *tree* against a searching victim; our frozen-champion case is the
  degenerate single-*line* version).

Collapse3's contribution is **not** the existence of the floor; it is the *exact
measurement* of it in a natural (not adversarially constructed) game where the
underlying MDP is fully solved:

- an **enumerated exact** floor for the single-hide observations (not a bound —
  see the Exactness note in Finding 4), rather than a worst-case guarantee;
- its **growth with game size** (0.0805 → 0.1677 from (4,4) to (5,5));
- **constructive memory recovery** to a **0.0000** floor from the observed
  destroyed-bead ledger (Finding 7);
- and the **on-policy null**: the provable floor is *behaviorally invisible* in
  actual play — the agents' real interface (the legal-move mask) erases most of it
  and their trajectories steer around the rest (Finding 7, and the interface
  ladder in the preamble).

Two observations about how our setting relates to Littman's (these are notes on
*this project*, not claims drawn from the cited papers):

- **Decomposability is why we can enumerate at all.** His performance measure
  (steps to goal) is *trajectory-coupled* — the state-visit distribution depends
  on the policy — which is what makes optimizing the memoryless policy NP-hard.
  Our floor is a *decomposable per-decision* regret with **fixed**
  (uniform-over-states) group weights, so each observation group optimizes
  independently and the whole thing enumerates in polynomial time. The coupled
  quantity Littman studies reappears here as the *on-policy* regret of Finding 7.
- **The floor binds any memoryless policy, not just deterministic ones.** Within
  an observation group, expected regret is a **linear** function of the action
  distribution *restricted to the actions legal in every member* (the group's
  feasible sub-simplex), and a linear function on a simplex attains its minimum
  at a vertex — a deterministic action. This relies on the weights being fixed;
  it does **not** contradict Littman's stochastic-beats-deterministic result,
  which lives in the trajectory-coupled setting where the visit distribution
  depends on the policy and the objective is no longer linear.

Finding 17 (what passing an evaluation rules out) sits in a fourth line — the
**methodology of evaluation itself**. It is a synthesis of mature ideas, not a
new method; the honest, per-area inventory with novelty verdicts is
[`EVALUATION_EQUIVALENCE_PRIOR_ART.md`](EVALUATION_EQUIVALENCE_PRIOR_ART.md). The
nearest neighbours:

- **Worst-case / exploitability.** Johanson, Waugh, Bowling & Zinkevich (2011),
  *Accelerating Best Response Calculation in Large Extensive Games* (IJCAI), made
  it feasible to measure a strategy's exploitability (worst-case value) in poker,
  and found top agents beat each other by tiny margins yet spanned a wide range of
  exploitability — win rate ≠ worst case. Bowling, Burch, Johanson & Tammelin
  (2015), *Heads-up Limit Hold'em Poker is Solved* (Science), pushed the same
  best-response yardstick to a certified bound. Finding 17's compatible outcome
  range is a best-response worst-case, but taken over the *set of policies an
  evaluation cannot distinguish* rather than one fixed strategy.
- **Oracle-graded move quality.** Guid & Bratko (2006), *Computer Analysis of
  World Chess Champions* (ICGA Journal), grade human play by centipawn loss
  against an engine — per-decision, opponent-independent grading, the ancestor of
  this repo's value-regret. Finding 17 asks what such grading, restricted to the
  states one opponent induces, fails to certify.
- **Elo critiques.** Balduzzi, Tuyls, Perolat & Graepel (2018), *Re-evaluating
  Evaluation* (NeurIPS), show Elo bakes in transitivity and is inflatable by
  redundant agents, and propose Nash averaging. Finding 10 is the exact-game
  version of the critique; Finding 17 explains *why* a single strong opponent
  under-certifies (it never visits the danger zone).
- **Diagnostic / behavioural suites.** Ribeiro, Wu, Guestrin & Singh (2020),
  *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList* (ACL), and
  Osband et al. (2020), *Behaviour Suite for Reinforcement Learning* (bsuite,
  ICLR), argue held-out accuracy overstates competence and push targeted,
  capability-level probing — the same move as reporting per-criticality regret and
  choosing the opponent that scrutinizes the states that matter.
- **Adversarial policies.** Gleave et al. (2020), *Adversarial Policies:
  Attacking Deep RL* (ICLR), and Wang et al. (2023) above: a candidate that scores
  well is beaten off-distribution. Finding 17 turns that phenomenon into an
  evaluation-design statement (use suboptimal / all-legal probing).
- **Sandbagging.** van der Weij, Hofstätter, Jaffe, Brown & Ward (2024), *AI
  Sandbagging: Language Models can Strategically Underperform on Evaluations* —
  strategic underperformance on capability evals. Finding 9 is the exact-game
  proof that value-regret audits *thrown value, not thrown intent*; Finding 17
  bounds how much a passing candidate can still hide.

---

## Why this matters for AI research

- **Evaluation.** Win rate and aggregate accuracy are opponent-dependent and
  criticality-blind; both systematically overstate competence. Report
  regret distributions per named opponent distribution, and condition on
  decision-criticality.
- **Representation vs. search.** A lossy *interface* imposes a floor no amount
  of training can cross — and here that floor grows with problem size, is
  priced exactly per interface (observation only, +legal-move mask, +memory),
  and is directly measurable, not anecdotal.
- **Deployment.** A frozen open-loop plan is a best-response to one line, not a
  strategy; it can lose to strictly worse play. Robustness requires closing the
  loop — re-solving, online adaptation, or a precomputed contingent policy —
  not just optimality on-distribution.

---

## Is this just undertraining? Would a bigger model help?

No — and it's worth being precise about why.

- **The floor lives in the input, not the optimizer.** If two genuinely
  different positions collapse to the same observation but need different optimal
  moves, then *any* memoryless policy over that observation — tabular, deep net,
  or a learned value head — is wrong on at least one. You cannot optimize below a
  bound carried by the input. This claim rests on the **enumeration** (Finding
  4), not on training curves — deliberately so, because training curves cannot
  demonstrate it here:

  > **Correction (external review).** An earlier version of this bullet cited
  > "a 1×–8× budget sweep that never closes the gap" as empirical confirmation.
  > That was unsupported — the shipped results file contained no sweep — and
  > when actually run (`representation_amplification.py`, now recorded), the
  > sweep shows on-policy regret *falling* with budget: 0.0033 → 0.0003 at
  > 1×–8× (60K–480K episodes). This does not contradict the floor; it is the
  > floor working as priced. The trained agent's interface is **mask-aware**
  > `hide_cooldown`, whose exact floor at (4,4) is **0.0000** (Finding 4's
  > interface table), and its audit runs on the states it actually visits,
  > where the matched floor is ≈0 (Finding 7's steering). Scaling closes the
  > gap *to the agent's own interface floor* — which is zero here. A budget
  > sweep can only exhibit the wall for an agent whose interface floor is
  > materially positive (e.g. mask-blind hide-reserves, floor 0.0805); the
  > wall itself is proven by enumeration either way.
- **Anchor the claim to the floor, not a loss percentage.** Trained loss rates
  are budget-sensitive and wander; the enumerated floor is stable and exact.
  Quote the floor.
- **Search with a true model doesn't count — and neither does the legal-move
  list.** An agent that expands the real game tree reads reserves/cooldown
  during search and de-aliases itself; more subtly, even the **legal-action
  mask** leaks the hidden fields (removals reveal cooldown; missing placements
  reveal an empty reserve — see the interface note in Finding 4). The clean
  test is a model-free agent restricted to the lossy observation alone; every
  wider interface has its own, smaller, exactly-priced floor.
- **Memory would help — and enumeration proves it.** A history-conditioned agent
  can reconstruct dropped fields from observed removals; the memory-augmented
  floor is **0.0000** at (4,4) (Finding 7). Trained agents may pay almost none of
  the memoryless floor on-policy — partly because their real interface (the
  legal-move mask) already erases most of it, partly because their trajectories
  steer around the rest (Finding 7).

---

## Rule sensitivity (a caution)

These results are specific to the shipped rule. A prototype used a "pass" rule
instead of immediate game-end; under it the same census reported a
`hide_cooldown` floor of 0.0137 and only ~40% draws. Changing one tiebreaker
ruling **moves the load-bearing hidden feature** (cooldown ↔ reserves). Claims
about "which feature matters" are fragile to the exact rules — a caution worth
stating explicitly in any write-up.

---

## Limitations & scaling

- **Exact enumeration ceiling.** (5,5) is 12.7M states (~26 min, CPU-bound).
  (6,6) would be ~100M+ states — beyond practical exact *enumeration*. The exact
  scaling curve for aliasing floors is therefore established through (5,5).
- **Rules correctness.** Every theorem rests on one engine's semantics. A
  clean-room reimplementation (`collapse3/reference_engine.py`, written from
  ``rules.md`` only) is differential-tested against the shipped engine:
  full agreement on all ~97K reachable states at (3,3), plus random deep
  playouts at (4,4) (and (5,5) under ``COLLAPSE3_SLOW=1``). Disagreement
  fails the default test suite.
- **The full game root is solved; enumeration is not.** The (14,14) opening is
  a first-player forced true win (+100), certified in ~96K nodes (Finding 11).
  Solving the root is not enumerating the game — aliasing floors, census, and
  misère still require visiting all reachable states and remain capped at (5,5).
  But every real full-size game is exactly gradeable (`experiments/horizon.py`).
- **First-move advantage** is exact, not incidental: the player to move is never
  worse than a draw at any reserve split through (5,5) (Finding 6), and from
  (6,6) through (14,14) the mover forces a true win (Finding 11), so
  head-to-head results are reported by seat. Learned-agent findings are at full
  training budget (150K episodes) at (4,4); pushing them to (5,5) is
  straightforward but slower.
- **The equal game is decisive at scale.** `(r, r)` is a draw for r = 1..5 but a
  first-player line win from (6,6) through (14,14) (Findings 6 and 11), and the
  draw fraction falls **monotonically** 85.8% → 71.5% → 54.0% → **37.7%** from
  (2,2) through (5,5). The game gets *sharper*, not drawish, as material grows
  — scale reserves, not rules, if you want fewer draws.

---

## Reproducibility

Every experiment records provenance via `experiments/_provenance.py`. Exact
claims are guarded by **reproduce-or-abort** tests: if a recomputed number
drifts from the shipped gate, the suite fails rather than quietly rewriting
the finding. Cite the `results/*.json` file for any number, not prose.

```bash
python -m experiments.aliasing_floor 5              # scaling floors (heavy)
python -m experiments.frozen_plan_vs_blunder 4      # inversions
python -m experiments.optimal_move_rate             # flattering aggregates
python -m experiments.elo_tournament                # Elo vs regret vs exploitability
python -m experiments.elo_tournament --bootstrap    # CI on the Elo inversion (from recorded H2H)
python -m experiments.structural_census 4           # where difficulty lives
python -m experiments.oneply_vs_learned 4           # performance vs competence
python -m experiments.objective_failure 4           # objective vs representation failure
python -m experiments.opening_values 6              # first-mover advantage + (6,6) flip
python -m experiments.depth_sweep                   # competence-per-ply (strategy ladder)
python -m experiments.best_response 3 4 5           # certify/refute frozen policies (Gate C)
python -m experiments.kimi_gates                    # distributional eval of the rulebook (Gates A/B)
python -m experiments.kimi_census                   # rulebook graded on every decision state
python -m experiments.memory_floor 4                # (4,4) baseline
python -m experiments.memory_floor 4 --opponent random
python -m experiments.memory_floor 5 --sweep          # (5,5) multi-seed/budget (heavy)
python -m experiments.sandbagging 3 4 5             # throwing is provably hard (misere solves)
python -m experiments.masked_floor 3 4 5            # floors when the legal-move mask is visible
python -m experiments.full_game_value               # 14×14 opening grid + root solve
python -m experiments.horizon                       # full-size game grading battery
python -m experiments.threepeg_floor                # Three-Peg sibling: full floor curve (2,2)-(14,14)
python -m experiments.sixpeg_floor                  # Six-Peg sibling: boundary moves with capacity (2,2)-(7,7)
python -m experiments.threepeg_selfplay             # self-play: near-perfect on-policy, global edge erodes, trips the boundary
python -m experiments.weak_exploiter                # a near-random policy (6 memorized moves) forces a >99%-vs-random champion to lose
python -m experiments.interface_ladder 4            # floor is a property of the interface (mask-blind->aware->memory)
python -m experiments.info_policy 3 4               # value of information: when to acquire a missing feature (sparse VOI)
python -m experiments.evaluation_equivalence 3 4 --full  # what passing an eval rules out: compatible outcome range (Gate C/A/D/B)

# Finding 16 — torch-dependent exhibit, quarantined in probes/ (needs: pip install numpy torch)
python probes/make_memo44.py                        # exact labels for every (4,4) state -> memo44.pkl (git-ignored)
python probes/generalization_experiment.py          # train-fraction sweep + cross-size + (5,5) extrapolation
python probes/chatgpt_train.py                       # train setup-A net, held-out audit, best-response certification
python probes/chatgpt_audit.py                       # recompute audit + certification from the saved checkpoint
python probes/matrix_reconcile.py                    # 6 nets (A/B x 3 seeds) -> 12/12 forced loss + removal-only reconciliation
```

Cite the `results/*.json` file for any number, not prose. Exact state counts and
aliasing floors have been independently reproduced (1,357,963 states;
hide-reserves floor 0.0805 at (4,4)).
