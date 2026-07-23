# What passing an evaluation rules out

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
- **Gate A (minimum-mutation) is not yet run.** No minimum distinct-mutation
  count is claimed; the provisional pre-freeze depth-limited figures are *not*
  inherited — every depth-limited number above is recomputed under the stated
  inclusive horizon convention.
- **Exact exhibit, not field evidence.** (3,3)/(4,4), one frozen reference
  policy. The transferable claim is the negative one in
  [`RELEVANCE.md`](../RELEVANCE.md): the protocol *can* pass a catastrophic
  policy — shown here where the answer key exists.

## Reproduce

```bash
python -m experiments.evaluation_equivalence 3 4 --full   # Gate C (3,3)+(4,4), Gate D/B (3,3)
pytest tests/test_evaluation_equivalence.py               # reproduce-or-abort Gate 0 + record guard
```

Gate 0 reproduces the (2,2) and (3,3) anchors (state counts, frozen-policy
hashes, control outcomes, and the optimal-only/all-legal split) before any
headline runs; a mismatch aborts. All numbers here are bound to
[`results/evaluation_equivalence_latest.json`](../results/evaluation_equivalence_latest.json).
