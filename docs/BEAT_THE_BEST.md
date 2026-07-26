# The "Beat the Best" Fallacy

## Why the strongest opponent is not the strongest evaluation

*A plain-language on-ramp to [Finding 17](FINDINGS.md#17-passing-a-test-rules-out-less-than-you-think--the-strongest-player-is-not-the-strongest-tester).
The exact tables, gates, observation-level hierarchy, and limitations live there
and in [`EVALUATION_EQUIVALENCE.md`](EVALUATION_EQUIVALENCE.md); this page is only
the intuition. Every number below is exact and reproducible.*

A common intuition in AI evaluation:

> **If you want the strongest evaluation, test against the strongest possible opponent.**

It seems obvious. A stronger opponent is harder to beat, so testing against
stronger opponents should produce a better evaluation.

Finding 17 is an exact counterexample — *exact* in the strict sense. Collapse3 is
solved, so "how bad could a candidate still be, given everything this evaluation
observed?" is a quantity we **compute**, not one we estimate from samples.

The surprising part is not that weak opponents sometimes expose weaknesses.
Adversarial testing and distribution shift are well-known, and this repository
claims no novelty there. The stronger result is:

> **Narrowing the opponent to its best play can make the evaluation certify
> less — here, all the way down to certifying nothing at all.**

---

## The mechanism in one sentence

A perfect opponent never plays an objectively losing move. If a candidate's
weakness only becomes reachable *after* such a move, then a perfect opponent can
never steer the game there — so the evaluation never looks, and passing it tells
you nothing about that weakness.

Choosing an opponent is not only choosing a difficulty level. It is choosing
**which states will ever be scrutinised**.

---

## What was actually measured

Take a candidate, let it be evaluated, and pin it to reference play at every
decision the evaluation inspected. Then let an exact adversary drive it
everywhere else, and solve for the worst outcome still consistent with what the
evaluation saw — the **compatible outcome range**.

| board | opponent family | decisions inspected | worst outcome still compatible |
|---|---|---|---|
| (3,3) | optimal-only | 41 | **forced loss** |
| (3,3) | all-legal | 96 | draw ✓ |
| (4,4) | optimal-only | **10** | **forced loss** |
| (4,4) | all-legal | 447 | draw ✓ |

Read the (4,4) rows together. An evaluator that only ever faces **optimal**
opponents inspects **10 decisions** and rules out nothing: a candidate that
passes it perfectly is still a *certified forced loss*. The same protocol against
**all-legal** opponents inspects **447** and rules that loss out. The effect is
one-sided at (3,3) and **two-sided at (4,4)** — it grows with the board.

**And the surviving flaw can be tiny.** How few non-canonical decisions must a
candidate that passed the optimal-opponent evaluation still make, to be
force-losable? Solved exactly: **one**. A single memorised mistake is enough —
and that does not grow with the board, holding from (3,3) through the
**12.7-million-state (5,5)** game.

---

## Why this happens

### 1. Difficulty and diagnostic value are different properties

**Difficulty** is how hard an opponent is to beat. **Diagnostic value** is how
much an opponent reveals about the candidate. These are different properties, and
nothing makes them rise together.

### 2. Optimal play narrows the evaluation

As an opponent becomes more optimal it stops making objectively bad moves — and
those bad moves may be exactly what makes the candidate's weakness reachable. The
opponent gets stronger. The evaluation gets narrower.

### 3. The blind spot is structural, not statistical

This is not a "too few games" problem. If an opponent's optimality prevents it
from entering a region of the state space, then no number of additional games
against that same opponent family will ever test that region. More data repeats
the same coverage instead of expanding it.

What fixed it in the experiment was changing the opponent **family** — not the
sample size. Sampling did eventually get there against all-legal opponents, but
inefficiently: about **3,000** audited games to reach a certification that exact
coverage nails with **96** decisions.

---

## "Isn't a subset trivially weaker?"

Partly, and that part is conceded: of course inspecting *fewer* decisions can
certify less. The direction is structural, not a discovery.

The finding is **where the threshold falls**. Restricting to optimal-opponent
play is the natural, widely-recommended choice — "test it against strong bots" —
and it is precisely what drops the inspected set from *certifying a draw* to
*certifying nothing*. The trivial direction does not tell you that the sensible
default lands on the wrong side of the line, that it does so from both seats once
the board grows, or that the flaw left standing needs only a single wrong move.

---

## What this does *not* say

**It does not say "use weak opponents."** This is the most tempting misreading.
The winning family here is *all-legal* — every legal move available, which
**includes** optimal play. It is a strictly *wider* family, not a weaker one. The
lesson is to avoid **restricting** your opponents to optimal play, not to go find
bad ones.

**It does not say a cleverer evaluator is the answer.** We tested that and it
failed. Strategically-chosen coverage bought *nothing* over depth-matched
**random** coverage drawn from the same exposure universe: wherever strategic
selection certified, same-universe random certified too. It is the **universe**,
not the selection. Reported as a plain falsification of our own hypothesis.

**It is not a new method.** This is the exact, game-theoretic instance of
surviving-mutant reasoning from mutation testing and of an *identified set* from
partial identification. The contribution is making it exact against solved ground
truth, not inventing the idea — see
[`EVALUATION_EQUIVALENCE_PRIOR_ART.md`](EVALUATION_EQUIVALENCE_PRIOR_ART.md) for
the full inventory and the honest novelty verdict.

---

## The central distinction

> **The strongest competitor is not necessarily the strongest evaluator.**

Passing against the strongest opponent certifies behaviour only on the states
that opponent is capable of reaching. It certifies nothing about the states that
the opponent's own optimality prevents it from ever visiting.

So evaluation quality cannot be read off opponent strength. A strong evaluation
has to ask two independent questions:

1. **How difficult is the opponent?**
2. **What is the opponent structurally incapable of testing?**

Almost all of the attention in practice goes to the first.

---

## Summary

The "Beat the Best" Fallacy is the belief that success against the strongest
opponent gives you the strongest evaluation. Finding 17 is an exact
counterexample, separating **playing strength** from **evaluating strength**: a
stronger opponent can be harder to beat while revealing less.

The reason is structural. An optimal opponent systematically avoids the very
states that would expose the candidate, and no amount of further testing against
that same family recovers them. The fix is a wider opponent universe, not a
smarter or a weaker one.

> **A better competitor can be a worse evaluator.**

---

**Scope.** Established at the **transcript** level, where the evaluator observes
and constrains the candidate's action at each visited state. The weaker rungs —
grading, and win-rate or Elo only — are deliberately quarantined here as unsound
to model this way, because a policy can be pinned to the right *actions* while
faking those signals. Nothing in this document claims those weaker methods escape
the limitation; the repository treats them as strictly **less** informative. All
results are for the solved sizes (3,3)–(5,5); see
[`EVALUATION_EQUIVALENCE.md`](EVALUATION_EQUIVALENCE.md) for what transfers and
what does not.
