# Relevance — why a solved toy game is about your metrics, not your model

*The author's framing, for the record:* Collapse3 is a microscope — not a
blueprint of the failure modes of frontier or real-world systems. But even with
perfect ground truth, no single metric captures competence, robustness, and
intent at once — and the last, no metric over outcomes and move-values alone
can reach inside a flat optimum.

*One-line version: the toy is not a model of your system — it is a unit test of
your metric, run where the answer key exists.*

---

## Why should anyone who works on real AI systems care about a toy game?

*The wall isn't at the frontier of capability — it's at the frontier of what we
can certify.*

This is the right first question, and the standing objection to all
model-organism work. Here is the honest answer: two arguments, one inversion,
one concession, one boundary, and a test.

**1. The object of study is the evaluation method, not the agent.**
Toy agents tell you little about frontier models — agreed. But this repo's
subjects are win rate, Elo, self-play saturation, held-out accuracy, and
average regret: *the same mathematical procedures* used on real systems, not
scaled-down analogies of them. These metrics carry no printed domain
restrictions — field practice deploys them as if universally comparable — so a
single certified counterexample under ground truth is enough to break that
usage. When [Finding 16](docs/FINDINGS.md) shows a near-zero generalization
gap coexisting with a certified 6-ply forced loss, that is a property of **the
metric as deployed**, demonstrated in the one setting where the metric can be
checked against the answer key.

**2. The failure modes shown here are already documented in real systems.**
The direction of evidence runs the opposite way from the criticism. The
weak-exploiter result ([Finding 14.iv](docs/FINDINGS.md)) is not an analogy to
Wang et al.'s adversarial-policies attack on superhuman KataGo — it is the
same structure, which occurred in a deployed system *first*. Finding 16 is the
exact version of a decade of adversarial-ML results. Evaluator-blind reward
gaming has a literature of wild examples. (Not every finding has a documented
wild counterpart yet — the Elo inversion of Finding 10 is stated here as a
mechanism, not a matched case.) Collapse3 does not predict that these failures
exist; **reality already established that. Collapse3 is where they can be
studied with proofs instead of anecdotes** — exhaustive adversaries,
attack-independent certificates, exact regret.

**3. What transfers is precisely stated: negative results about metrics.**
The claim that travels is of the form *"this evaluation procedure can be fully
satisfied while the agent it endorses is exploitable"* — an existence proof
about the metric. What does not travel is any **quantity**: no floor,
threshold, or percentage here is a fact about any other system. The repo's own
sibling geometries prove the point — exact values fail to transfer even
between [its own variants](docs/FINDINGS.md).

**4. The concession, so the argument stays honest.**
Nothing here demonstrates that any particular real system has these failures,
or estimates how likely they are in practice. A solved 3×3×3 game cannot do
that, and this repo never claims it. The demonstrated claim is narrower and
harder to dismiss: **the metrics the field relies on are provably capable of
missing exactly these failures, because here they did — under ground truth,
with certificates.**

**5. We tested transfer where testing was possible — and it failed there.**
"Does a toy transfer to real systems?" is itself an evaluation-transfer
question, and evaluation transfer is this project's subject. External transfer
— toy to real — cannot be tested from inside a toy. But transfer across our
own sibling variants *can* be, and was: exact values deform or vanish between
geometries that differ by one rule. A project whose numbers don't survive the
trip next door does not ask you to carry them across the world — and it can
state, with certificates, exactly why not.

**6. The transfer test.**
A finding here licenses a claim about a real system when three conditions
hold, in order: **(i)** the real evaluation uses the *same procedure* — the
identical estimator, not an analogy; **(ii)** the structural preconditions
that triggered the failure here are measurable and present there — for the
Elo inversion, a capped upside for optimal play and a heterogeneous opponent
pool; for Finding 16, a nonzero error rate and an adversary able to steer
toward reachable errors; for the sandbagging exhibit, a flat optimum where
choices among value-equal moves go unscored by the auditor; **(iii)** a
ground-truth or best-response audit in the real system actually confirms the
divergence. If (i)–(iii) all hold, the finding has carried over, not merely
been invoked. If only (i) holds, what travels is the negative claim about the
metric — nothing more. We would rather a reader apply this test and reject a
transfer than repeat our result as an analogy. **Analogies are how toy results
get over-read; this test is how they get used.**

---

*One-line version: the toy is not a model of your system — it is a unit test of
your metric, run where the answer key exists.*
