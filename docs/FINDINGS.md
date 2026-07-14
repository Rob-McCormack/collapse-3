# Findings & Significance

All numbers below are produced by the scripts in `experiments/` and written to
`results/*.json` with the git commit, config, and seed that generated them.
Figures are for the shipped rule (**immediate game-end on no legal action**);
exhaustive results are exact, and any illustrative small-budget run is labelled.

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
scalar transfers.

We deliberately use *value* regret, not policy distance `|π_agent − π_opt|`:
optimal play here is highly non-unique (many moves hold a draw), so policy
distance would wrongly penalize correct-but-different moves.

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

### 4. Representation cost is real, quantifiable, and scales
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

The reserves-hidden floor grows steeply and monotonically (≈2× from (4,4) to
(5,5)); the cooldown-hidden floor stays tiny and plateaus. So the cost of a
missing state feature is **structural and grows with the game**, not an artifact
of small-game triviality — and reserves, not cooldown, are the load-bearing
feature. The novel content here is not that such a floor *exists* (that is
classical — see [Relation to prior work](#relation-to-prior-work)); it is that
we can **enumerate it exactly**, watch it **grow with game size**, and — for the
single-hide columns — show it is the *exact* optimum, below.

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
the count, leaving a real residual — note the masked reserves floor no longer
grows with size the way the mask-blind one does. (iii) The headline numbers
(0.0805, 0.1677) are therefore **exact for the mask-blind interface, not lower
bounds on the repo's own trained agents**, whose interface is mask-aware. The
honest one-line reading: *how much a missing feature costs depends on the
interface it is missing from — and Collapse3 can price each interface exactly.*

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

### 7. The floor is real, but on-policy regret can steer around it
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

**Conclusion.** The aliasing floor is **provable yet behaviorally invisible** at
every tested size and opponent distribution: uniform floor **0.0805 → 0.1677** as
the game grows, but trained on-policy regret stays ≪ floor because policies steer
which aliased states they visit. Realized regret is a property of **input and
trajectory**, not the floor alone. Independent reimplementation matched **1,357,963**
states and **0.0805** at (4,4) to four decimals.

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
enumerated sizes* ((3,3)–(5,5)); the 14-bead game is unsolved and open. The
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
the tournament was played at — out-rates the optimal agent 1825 to 1804, while
the right-hand columns tell the truth: optimal is the *only* agent with zero
losses and zero regret, and its worst case is a theorem where the
table-topper's is a 7-ply kill.

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
"who plays correctly" — and in a solved game the two visibly disagree.
Accessible version in [FAQ #10](FAQ.md); numbers in
[`results/elo_tournament_latest.json`](../results/elo_tournament_latest.json).

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

Collapse3's contribution is **not** the existence of the floor; it is the *exact
measurement* of it in a natural (not adversarially constructed) game where the
underlying MDP is fully solved:

- an **enumerated exact** floor for the single-hide observations (not a bound —
  see the Exactness note in Finding 4), rather than a worst-case guarantee;
- its **growth with game size** (0.0805 → 0.1677 from (4,4) to (5,5));
- **constructive memory recovery** to a **0.0000** floor from the observed
  destroyed-bead ledger (Finding 7);
- and the **on-policy null**: the provable floor is *behaviorally invisible* in
  actual play, because trained policies steer around the aliased states.

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
  floor is **0.0000** at (4,4) (Finding 7). Trained agents may still pay almost
  none of the memoryless floor on-policy if they steer around aliased states.

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
  (6,6) would be ~100M+ states — beyond practical exact solving. The exact
  scaling curve is therefore established through (5,5).
- **Rules correctness.** Every theorem rests on one engine's semantics. A
  clean-room reimplementation (`collapse3/reference_engine.py`, written from
  ``rules.md`` only) is differential-tested against the shipped engine:
  full agreement on all ~97K reachable states at (3,3), plus random deep
  playouts at (4,4) (and (5,5) under ``COLLAPSE3_SLOW=1``). Disagreement
  fails the default test suite.
- **Toward the real game (14 beads).** Requires an **approximate oracle**
  (e.g. strong MCTS) with bounded error, or a solver-generated curriculum of
  critical states. The exact curve motivates the hypothesis; the approximate
  study would test whether it persists at full scale.
- **First-move advantage** is exact, not incidental: the player to move is never
  worse than a draw at any reserve split (Finding 6), so head-to-head results are
  reported by seat. Learned-agent findings are at full training budget (150K
  episodes) at (4,4); pushing them to (5,5) is straightforward but slower.
- **The equal game is decisive at scale.** `(r, r)` is a draw for r = 1..5 but a
  first-player line win at (6,6) (Finding 6), and the draw fraction falls
  **monotonically** 85.8% → 71.5% → 54.0% → **37.7%** from (2,2) through (5,5).
  The game gets *sharper*, not drawish, as material grows — scale reserves, not
  rules, if you want fewer draws.

---

## Reproducibility

Every experiment records provenance via `experiments/_provenance.py`:

```bash
python -m experiments.aliasing_floor 5              # scaling floors (heavy)
python -m experiments.frozen_plan_vs_blunder 4      # inversions
python -m experiments.optimal_move_rate             # flattering aggregates
python -m experiments.elo_tournament                # Elo vs regret vs exploitability
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
```

Cite the `results/*.json` file for any number, not prose. Exact state counts and
aliasing floors have been independently reproduced (1,357,963 states;
hide-reserves floor 0.0805 at (4,4)).
