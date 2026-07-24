# What passing an evaluation rules out

> *We don't evaluate agents. We evaluate the distribution of states a specific
> opponent induces. If the opponent isn't permitted to play badly, we're relying
> on it to wander into the danger zone by chance — and a perfect opponent cannot
> wander there at all. The states that would expose the weakness are exactly the
> ones optimal play refuses to enter.*

*After a candidate **passes** an evaluation, how bad can a policy still be while
remaining consistent with everything that evaluation observed? Here the question
has an exact answer — and the answer is "much worse than you'd think, and it is
the **opponent's suboptimal moves**, not the opponent's strength, that close the
gap."*

> **Scope — read this first.** This is a **transcript-level** result (defined
> below) at reserves **(3,3)** and **(4,4)**, computed exactly on the shipped
> pure-Python engine — no neural nets, no sampling in the headline. It is an
> **exact exhibit**, not evidence about any deployed system; nothing here is a
> claim about frontier evaluation beyond the negative, existence-proof form
> ("this protocol *can* pass a catastrophic policy, because here it did"). Every
> number is bound to
> [`results/evaluation_equivalence_latest.json`](../results/evaluation_equivalence_latest.json)
> and guarded by `tests/test_evaluation_equivalence.py`.

---

## The quantity

An evaluation constrains a candidate's actions at a set of states **C** (the
states it actually inspected). A policy is **benchmark-compatible** if it agrees
with the reference wherever the evaluation looked. Over *all* compatible
policies we report the **compatible outcome range** `(worst, best)`:

- **worst compatible** — the most dangerous agent that could still have posted a
  perfect record. All the scientific content lives here.
- **best compatible** — the highest value any compatible policy can reach. With
  an optimal reference this is *always the game value* (the reference is
  compatible with itself and optimal), so it is an **assertion, not a finding**:
  it is checked at every configuration and, if it ever failed, the solver would
  be wrong.

It is computed **exactly** by best-response with **pinned nodes** — not policy
search. Inside `C` the mover is forced to the reference action; outside `C` the
adversary drives the candidate (worst case) or the candidate plays for itself
(best case). One minimax pass each. We report `worst`, `best`, `identified`
(whether the two coincide) — never a scalar "width", because WDL outcomes are
**ordinal** and a numeric difference would imply a cardinal scale the units do
not carry.

## Observation levels

An evaluator is defined by which states it visits **and what it observes there**.
Three rungs, decreasing information; the compatible policy class grows at each:

| rung | observes | examples | status |
|---|---|---|---|
| **transcript** | the action itself | state audits, adversarial coverage | **implemented** |
| **grade** | only that the action was zero-regret | oracle-graded play | quarantined |
| **outcome** | only terminal results | win rate, Elo | quarantined |

Only the transcript rung is modelled here. The other two are deliberately left
as `NotImplementedError`: modelling them by pinning encountered actions would
grant the evaluator information it never had and **overstate** its certification
power — understating this repo's own thesis. (A one-pass pinned solve is
provably unsound for `grade`: a compatible policy can deviate within `C`, escape
into states `C` never covered, and misbehave there, while a real evaluation
would have *followed* it. The correct form is a fixpoint over the policy's own
reachable support.)

## The headline: the strongest player is not the strongest tester

**Gate C** pins the *complete* benchmark support — every candidate decision the
protocol can ever expose — under two opponent families: **optimal-only** (the
opponent plays only value-optimal moves) and **all-legal** (the opponent may
play any legal move). Candidate follows the frozen reference `CANONICAL_POLICY_V1`.

| size | seat | optimal-only | all-legal |
|---|---|---|---|
| (3,3) | 0 | 41 pinned → **forced loss** compatible | 96 pinned → draw (identified) |
| (3,3) | 1 | 367 pinned → draw | 538 pinned → draw |
| (4,4) | 0 | **10** pinned → **forced loss** compatible | 447 pinned → draw (identified) |
| (4,4) | 1 | 1,241 pinned → **forced loss** compatible | 3,982 pinned → draw (identified) |

Read the (4,4) seat-0 row slowly. An evaluator that tests the candidate against
**only optimal opponents** inspects **10 decisions** and certifies nothing: a
policy that passes it perfectly can still be a **certified forced loss**. The
*same* protocol against all-legal opponents inspects 447 decisions and rules the
loss out. The effect is **seat-0-only at (3,3) but two-sided at (4,4)** — it
grows, not shrinks, with the board.

**Why it is not merely "fewer constraints certify less."** That direction is
structural: the optimal-only support is a subset of the all-legal support, so it
can only certify less-or-equal. The *finding* is that restricting the opponent
to **optimal play** — the most natural "test against strong bots" choice — is
exactly what shrinks the inspected set below the certification threshold. A
perfect opponent *refuses to enter objectively losing lines*, and one of those
refused lines is precisely where the candidate's specific weakness lives. So:

> **Opponent playing-strength and evaluator strength are distinct properties.**
> Testing only against strong opponents can leave catastrophic,
> candidate-specific weaknesses completely untested — the states that would
> expose them are the states a strong opponent will never visit.

This is the same mechanism as the repo's adversarial-policy results: the
best-response adversary in [Finding 16](FINDINGS.md) and the weak exploiter in
[Finding 14](FINDINGS.md) both win by playing **true-game blunders** the trained
agent never had to answer. Here it is stated as a property of the *evaluation
protocol* itself, exactly.

## H2 falsified: it is the universe, not the selection

A tempting follow-on hypothesis (**H2**): *strategic* depth-limited coverage
should certify more than *random* coverage of the same size. **Gate D** tests it
against two depth-matched random controls — control **A** draws from all
decision states; control **B** (the strong one) draws from the same all-legal
exposure universe, matching the exact ply-depth histogram, so only the selection
rule differs. (Depth is the exact invariant `plies = 2·placed − on_board`,
verified against BFS with zero mismatches; the horizon convention is inclusive
and stamped into the record.)

At (3,3), 30 seeds per cell (`force-losable` = seeds where the random set still
left a forced loss):

| seat | budget | strategic | control A (all states) | control B (same universe) |
|---|---|---|---|---|
| 0 | 4-ply | draw (certifies) | 30/30 losable | **0/30 losable** |
| 0 | 6/8-ply | draw | 30/30 | 0/30 |
| 1 | 6-ply | draw (certifies) | 30/30 | **0/30 losable** |
| 1 | 4-ply | forced loss | 30/30 | 30/30 |

Whenever strategic coverage certifies, **same-universe random coverage certifies
too** (control B 0/30); whenever strategic fails, control B fails as well.
Strategic selection buys **nothing** over depth-matched random *from the same
universe*. What certifies is drawing from the **all-legal exposure universe** at
all — control A, matched on depth but drawn from the whole state space, never
certifies. **H2 is falsified at the budgets tested**, and reported as such: this
does not prove strategic selection can never help at some smaller budget, only
that at these budgets the universe, not the selection, is what matters.

## Gate B: finite games are an inefficient way to cover

**Gate B** replaces exhaustive coverage with `n` transcript-audited games
(noisy opponent). Certification (draw) is reached only at **3,000 games**
(seat 0: 91 decisions / 94.8% of exposure; seat 1: 485 / 90.1%). Below that a
forced loss survives at every budget. Targeted all-legal coverage certifies the
same seats with **96 / 538** decisions exactly — so playing thousands of games,
the way agents are usually evaluated, approaches a certification that exact
coverage reaches directly. (This is a first data point toward the untested H4,
not a closed result.)

## Gate A: how weak can the surviving flaw be?

Gate C shows a benchmark-passing policy *can* be force-losable. **Gate A** asks
the minimum-cardinality question: how *few* non-canonical decisions must such a
policy make? A candidate is pinned to the reference on the optimal-only support
(it passed the perfect-opponent evaluation); Gate A then finds, by exact
lexicographic shortest-path (forced-loss → fewest candidate **mutations** →
least depth), the smallest number of decisions it can play non-canonically and
still be driven to a certified loss.

**The answer is one — and it does not grow with the board.** At (3,3) seat 0 a
single non-canonical decision (first reachable at depth 6) is enough: the policy
agrees with the reference everywhere the perfect opponent ever looked, deviates
in exactly one place a *suboptimal* opponent can steer it into, and loses by
force. It stays **one mutation, from *both* seats, all the way to (5,5) — a
12.7-million-state board.** (At (3,3) seat 1 is not force-losable under this
support, consistent with Gate C.)

| size | states | seat 0 | seat 1 |
|------|--------|--------|--------|
| (3,3) | 97K | 1 mutation (depth 6) | not force-losable |
| (4,4) | 1.36M | 1 mutation (depth 6) | 1 mutation (depth 5) |
| (5,5) | 12.7M | 1 mutation (depth 6) | 1 mutation (depth 5) |

This is the exact, minimum-cardinality cousin of the weak-exploiter result
([Finding 14](FINDINGS.md)): the surviving flaw a strong-opponent test misses is
not merely small, it is a **single memorized mistake**, and growing the board
~130× across three exact sizes did not make the candidate need more of them.
(Full Gate C is memory-bound at (5,5); Gate A there needs only one solve and one
Dijkstra, so it is recorded via `--ga5`.)

## What is *not* established

- **`best == game value` is an assertion, not a finding** — a solver sanity
  check, true in every configuration by construction.
- **The subset direction is structural.** Only the *threshold crossing*
  (optimal-only coverage falling below the certification line, two-sided at
  (4,4)) and its interpretation are the result.
- **Transcript level only.** The `grade` and `outcome` (win-rate/Elo) rungs are
  quarantined, not solved; nothing here certifies what Elo or win rate rule out.
- **Units and reference are frozen and WDL.** `CANONICAL_POLICY_V1` is the
  WDL-optimal move first under an explicit semantic key `(action_rank, peg,
  index)`; its per-seat hash is stamped and regression-tested so a legal-move
  reordering cannot silently change the equivalence classes.
- **Gate A is exact at (3,3)+(4,4)+(5,5), not swept beyond.** The
  minimum-mutation count (=1 wherever a loss is compatible) is reported only
  where it is exactly computed; (6,6)+ is not enumerable. Three exact points are
  corroboration, not an asymptotic claim (the sibling-geometry non-transfer of
  [Finding 4](FINDINGS.md) is the standing caution). The provisional pre-freeze
  depth-limited figures are *not* inherited — every depth-limited number above is
  recomputed under the stated inclusive horizon convention.
- **Exact exhibit, not field evidence.** (3,3)/(4,4), one frozen reference
  policy. The transferable claim is the negative one in
  [`RELEVANCE.md`](../RELEVANCE.md): the protocol *can* pass a catastrophic
  policy — shown here where the answer key exists.

## Reproduce

```bash
python -m experiments.evaluation_equivalence 3 4 --full --ga5  # Gate C (3,3)+(4,4); Gate A (3,3)+(4,4)+(5,5); Gate D/B (3,3)
pytest tests/test_evaluation_equivalence.py               # reproduce-or-abort Gate 0 + record guard
```

Gate 0 reproduces the (2,2) and (3,3) anchors (state counts, frozen-policy
hashes, control outcomes, and the optimal-only/all-legal split) before any
headline runs; a mismatch aborts. All numbers here are bound to
[`results/evaluation_equivalence_latest.json`](../results/evaluation_equivalence_latest.json).

## Prior art

The compatible outcome range is **not a new method** — it is the exact,
game-theoretic instance of a *surviving mutant* (mutation testing) and an
*identified set* (partial identification; closest ML relative: Skalse et al.
2023 on reward identifiability), and the attack mechanism is the
adversarial-policy phenomenon (Gleave 2020; Wang 2023). What is new is the
*synthesis made exact* against a solved game's answer key, plus the
opponent-family result. Full inventory, per-area analogues, and the novelty
verdict: [`EVALUATION_EQUIVALENCE_PRIOR_ART.md`](EVALUATION_EQUIVALENCE_PRIOR_ART.md).
