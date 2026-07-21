# Generalization is not robustness

*A trained network can ace states it never saw — and still be force-losable in
five or six plies. Both facts are exact. The first is measured against the
solver; the second is certified by it.*

> **Scope — read this first.** Every agent in this document is a **trained
> neural policy**; nothing here is a fact about Collapse3's solved values, and
> none of it may be quoted as such. Training and grading are at **(4,4)** unless
> marked otherwise; the policies are small MLPs on raw board features, not
> frontier models. This is an **exact exhibit, not field evidence** about any
> deployed system. All headline numbers were reproduced in-repo with torch
> 2.8.0 (CPU) and are bound to the results JSON cited at the foot of each
> section; **deterministic** quantities (state counts, the 12/12 certification,
> the reconciliation structure) match exactly, while **neural** decimals and the
> exact forcing line drift with torch build. Numbers marked *reported* come from
> the originating run and await the same in-repo re-audit. Random train/test
> splits leak child values at high train fractions and are not symmetry-grouped
> — see Limitations.

---

## The question

Machine learning's most trusted inference is:

> It performs well on unseen data, therefore it generalizes.

This document shows that inference can be **completely true** and still tell us
almost nothing about worst-case robustness — in a setting where both sides of
the claim can be settled exactly:

- **Generalization asks:** how often are you right on unseen states?
- **Robustness asks:** can an adversary find the states where you are wrong —
  and make those states matter?

Collapse3 answers both with arithmetic, not argument: the solver grades every
move, and the best-response solver is the strongest adversary that exists.

## The setup

Two independent training setups, with different targets, encodings, and
architectures, so no result rides on one modelling choice:

| | setup A (WDL classifier) | setup B (raw regressor) |
|---|---|---|
| input | 59-dim signed occupancy | 86-dim one-hot |
| target | 3-class P0 WDL, class-balanced CE | raw value / 100, MSE |
| net | 59 → 128 → 128 → 64 → 3 | 86 → 256 → 256 → 1 |
| policy | expected WDL of children | predicted raw value of children |
| training | AdamW, 8 epochs | Adam, 6 epochs |

Common protocol: exact labels for all **477,960** (4,4) decision states;
seed-fixed splits (20% train / 20% held-out / 60% unused); three seeds
(314159, 0, 1). Grading is identical for every net — WDL-optimality and WDL
regret against the enumeration. Certification is the repo's own `best_response`
instrument (`experiments/best_response.py`): submit the frozen policy, receive a
proof. Immediate terminals are scored from the rules, not learned, so every
error the adversary exploits is a genuine judgement of the net.

## 1. The generalization gap is approximately zero

Setup A, seed 314159, held-out side verified in-repo to five decimals:

| split | optimal | critical optimal | WDL regret (critical) |
|---|---|---|---|
| train (95,592) | 0.9812 *reported* | 0.9540 *reported* | — |
| held-out (95,592) | **0.98026** | **0.95149** | 0.0216 (0.0532) |

The train/held-out gap is a fraction of a point. And it is not an artifact of
the leaky 20%-train split: the train-fraction sweep in
`probes/generalization_experiment.py` shows the net already at **0.933** optimal
(0.877 critical) on held-out states with only **0.5%** of states in training —
where child-value leakage is negligible — rising monotonically to 0.978 at 80%.
This is not a memorized lookup table. In a separate run the net even extrapolates
to a **larger game**: trained on (3,3)+(4,4) and graded on a 1,200-state (5,5)
sample it scores **0.923** optimal (0.908 critical). The network really did
generalize. That is what makes the next section interesting.

*Bound to `results/neural_best_response_latest.json` (held-out) and the
`probes/generalization_experiment.py` train-fraction sweep.*

## 2. The adversarial gap is a forced loss — certified, 12 of 12

Every trained policy, frozen and handed to the best-response solver from the
drawn (4,4) root:

| seed | setup | seat 0 | seat 1 |
|---|---|---|---|
| 314159 | A | forced loss, depth 6 | forced loss, depth 5 |
| 314159 | B | forced loss, depth 6 | forced loss, depth 5 |
| 0 | A | forced loss, depth 6 | forced loss, depth 5 |
| 0 | B | forced loss, depth 6 | forced loss, depth 5 |
| 1 | A | forced loss, depth 6 | forced loss, depth 5 |
| 1 | B | forced loss, depth 6 | forced loss, depth 5 |

Twelve certifications, twelve forced losses. A policy that plays ~98% optimally
on states it never saw is beaten, on demand, from **both seats**, by an
adversary that needs at most **six plies**. The *outcome* (forced loss) is the
attack-independent fact; the exact *depth* is a property of the particular
trained weights and drifts with torch build — in this reproduction every
certificate lands at depth 6 (seat 0) / depth 5 (seat 1).

**Why this certificate is different from adversarial-ML practice.** "High test
accuracy, adversarially exploitable" is a decade-old phenomenon, and the repo
already cites its existence in the wild (the KataGo adversarial-policy result;
see [Finding 14](FINDINGS.md#14-self-play-saturates-flawless-on-its-own-trajectories-weaker-off-them-and-it-trips-over-the-phase-boundary)).
What the literature almost never gets is what this table is made of: robustness
claims there are **attack-relative** — certified against PGD, broken by a
stronger attack next year. Here the attacker is the best-response solver:
exhaustive, exact, the strongest adversary that exists. The certificate is
**attack-independent**, the generalization numbers are exact, and the exploit's
mechanism is fully traceable. The contribution is not the phenomenon; it is that
the phenomenon finally comes with a proof.

*Bound to `results/matrix.json` (all 12) and `results/neural_best_response_latest.json`
(seed 314159 lines).*

## 3. The exploit's anatomy

The certificates are not black boxes. Reading the seed-314159 / setup-A lines:

**The adversary wins by playing badly.** In the seat-1 certificate the
adversary's ply-3 move is *itself a true-game blunder* (`wdl −1`) — a move that
loses value against perfect play but wins against this particular net; in the
seat-0 certificate the adversary's ply-2 move is the deliberate blunder. This is
the organic inversion of
[Finding 3](FINDINGS.md#3-an-open-loop-plan-is-not-a-strategy--shown-exactly):
the exploit only exists because the opponent is *not* the oracle. A policy can be
near-perfect against the distribution and worthless against an opponent willing
to be wrong on purpose. (The net's own decisive error follows — seat 0 at ply 5,
seat 1 at ply 2.)

**The adversary finds whichever error exists.** Across the 12 certifications the
first value-losing move lands variously in the **held-out (test)** split and the
**unused-60%** split — both are generalization failures on states the net never
trained on, and which one appears varies by seed. (In this reproduction none of
the 12 first errors fell in the train split; Kimi's originating run reported a
train-split first error too — an expected run-to-run difference. The point is
split-agnostic: the adversary optimizes over the *whole* error set and does not
care which kind of error the net has.)

**Why the two gaps diverge, stated structurally.** Average-case competence
means: *the error set is small.* Worst-case robustness in a drawn game means:
*the error set is empty, or unreachable.* Those are not adjacent properties —
they are separated by the difference between 2% and 0. A 2–5% critical error
rate over ~478K decision states is thousands of candidate error states; the
best-response solver only needs **one** of them to be force-reachable. Small is
not empty, and an adversary converts "small" into "systematically exploitable"
precisely because it is searching for exactly that. Nothing about a low
generalization gap even gestures at this quantity.

*Bound to the `line`/`grading` blocks in `results/neural_best_response_latest.json`
and the `cert.first_error_state` fields in `results/matrix.json`.*

## 4. The failure anatomy is itself setup-dependent

*Included so this document does not overclaim in the other direction.*

An earlier analysis (setup B, 80% train) found errors concentrating on
**removal-only-optimal** critical states and concluded the game's cascade
mechanic was the irreducible core for learned approximators. The reconciliation
matrix — both setups, both definitions of the removal-only set, three seeds,
identical grading — refutes it. Critical-state optimal rate:

| seed | setup | removal-only critical | other critical |
|---|---|---|---|
| 314159 | A | **0.9917** | 0.9480 |
| 314159 | B | 0.9336 | 0.9231 |
| 0 | A | **0.9879** | 0.9508 |
| 0 | B | 0.9535 | 0.9285 |
| 1 | A | **0.9885** | 0.9512 |
| 1 | B | 0.8840 | 0.9303 |

The two definitions of "removal-only" (WDL-optimal vs raw-optimal) pick nearly
identical sets (~3.2K states, differing by 3–4) and change nothing. The training
*target* changes everything: the WDL classifier finds removal-only states
*easier* than other critical states, on every seed; the raw regressor is
unstable there (down to 0.884). Neither "the cascade is the irreducible core"
nor its negation is a fact about the game — both are facts about metric and
target choice. The anatomy of a learned policy's errors is setup-dependent all
the way down; the robust quantity is the certificate in §2, not the error
description. (Hypothesis, labelled as such: removal-only states are tactically
sharp but categorical — remove or lose the outcome — which a classification head
handles well, while fine positional discrimination among placements is the
harder judgement.)

*Bound to `results/matrix.json`.*

## What this is, and what it is not

The repo's existing thesis is that bad metrics mislead: win rate, Elo, on-policy
regret, and averages can hide weakness
([Finding 1](FINDINGS.md#1-aggregate-metrics-are-adversarially-flattering),
[Finding 10](FINDINGS.md#10-elo-prefers-the-exploitable-agent--the-rating-inversion-measured)).
This document extends the same thesis to the **best** metric — held-out
accuracy, distribution controlled, gap ≈ 0 — and shows it still does not certify
worst-case behavior. One pillar, deeper foundation:

> **Performance is not competence. Generalization is not robustness.**

For AI evaluation the lesson is pointed: "it generalizes" and "it is safe to
deploy against an adversary" are different claims, and the gap between them is
not estimable from any average-case number — here, ~98% on unseen states
coexisted with a certified forced loss, and no held-out audit of any size could
have found it, because the failure is not in the distribution. It is in what an
adversary can steer toward.

## Limitations

- Trained policies at (4,4), small MLPs on raw features; nothing about frontier
  systems is demonstrated — this is an exact exhibit, not field evidence.
- Random splits leak child values at high train fractions and are not
  symmetry-grouped; the headline certificate does not depend on the split, but
  the generalization numbers do (symmetry-grouped splits are the listed next
  step). The 0.5%-train row is included precisely because it is leakage-light.
- The (5,5) extrapolation test is a 1,200-state playout sample, not a census.
- Train-split accuracy in §1 is as-reported; the held-out side is verified
  in-repo. Direction of the gap is not in doubt; its second decimal is.
- The exact forcing depth and line are weight-dependent and drift with torch
  build; the *outcome* (12/12 forced loss) is the reproducible claim.

## Reproduction

All runs are in `probes/` (torch-dependent; quarantined from the zero-dependency
core). Chain: label generation (`make_memo44.py` → `memo44.pkl`), both training
setups (`chatgpt_train.py`, and setups A/B inside `matrix_reconcile.py`), the
exact held-out audits, the best-response certifications with exploit lines and
split provenance (`chatgpt_audit.py`), the reconciliation matrix
(`matrix_reconcile.py`), and the stdlib generalization sweep + (5,5)
extrapolation (`generalization_experiment.py`). Anchor numbers for every stage
are listed in the probes' handoff notes; deterministic anchors must match
exactly, neural decimals may drift with torch versions. To promote any number to
a repository result: re-run in-repo, stamp provenance, add a reproduce-or-abort
gate. **Cite the results JSON, not this prose.**
