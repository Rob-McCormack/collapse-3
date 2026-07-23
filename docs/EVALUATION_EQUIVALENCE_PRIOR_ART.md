# Prior art for the evaluation-equivalence result (Finding 17)

*An **inventory, not a verdict.** Finding 17's core object — "over all policies
consistent with what an evaluation observed, how bad can the worst one be?" — has
close relatives in at least four literatures that rarely cite one another. This
document catalogues them so a reader can judge the novelty claim honestly. The
repo has twice claimed ground that had parents; the safe default here is
**"new exact application / synthesis, not new method,"** and a human decides what
survives.*

The honest one-line placement: **the compatible outcome range is the
game-theoretic, ground-truth instance of a "surviving-mutant" / "identified-set"
argument.** Neither of those ideas is ours. What is new is (a) computing them
*exactly against a solved game's answer key* rather than estimating them, and
(b) the specific result that the **opponent family (the exposure universe), not
clever selection, is the lever** — "the strongest player is not the strongest
tester," with H2 (strategic beats random) falsified.

---

## 1. Mutation testing / test adequacy — the closest software analogue

**Closest analogue.** A test suite's power is measured by which *mutant* programs
it fails to distinguish from the original: a **surviving mutant** is a behaviour
the suite cannot rule out. "Worst compatible policy" is the direct game-theoretic
counterpart of a surviving mutant, and "identified" is a mutant being killed.

**Established by.** DeMillo, Lipton & Sayward (1978) introduced mutation testing
and the coupling effect; Jia & Harman (2011) survey three decades of it. The
*mutation score* (fraction of mutants killed) is a standard adequacy metric.

**What Collapse3 instantiates differently.** Mutation testing (i) enumerates
*syntactic* program mutants, not the *semantic* space of all policies consistent
with the observations; (ii) has no ground-truth oracle for "is this mutant
actually dangerous" — equivalent-mutant detection is undecidable in general;
(iii) reports a *score*, where we report an exact ordinal outcome (`worst`,
`best`, `identified`). We compute the *entire* compatible-policy extreme exactly,
against a solved-game answer key, with a proof.

**Novelty verdict:** *new exact application* of a surviving-mutant argument in a
setting where the "did the survivor actually matter" question is decidable.

## 2. Test-suite minimization / hitting-set formulations

**Closest analogue.** Choosing which decisions an evaluation must inspect to
retain certification power is a coverage/hitting-set problem; Gate A (minimum
mutation, still unrun) is explicitly a minimum-cardinality question.

**Established by.** Harrold, Gupta & Soffa (1993) frame test-suite reduction as
finding a minimal representative set covering all testing requirements, and note
the minimal hitting set is **NP-complete**.

**What Collapse3 instantiates differently.** Minimization preserves a *coverage
criterion*; we ask what a given coverage preserves about the *true worst-case
outcome*. Their objects are test cases and requirements; ours are game states and
a certified game value. Gate A's lexicographic objective (forced-loss →
min-policy-mutations → min-depth) is a specific exact instance, not a heuristic.

**Novelty verdict:** *new exact application* (shared combinatorial structure,
different quantity certified).

## 3. Partial identification / identified sets — the closest ML/statistics relative

**Closest analogue.** In econometrics, when the data plus assumptions do not pin
a single parameter, one reports the **identified set** of all parameter values
consistent with them. The compatible outcome range is an identified set over
*policies*, and `identified: bool` is exactly point- vs. set-identification.

**Established by.** Manski (1995, 2003) pioneered partial identification; Tamer
(2010) reviews it. The closest ML instance is **Skalse, Farrugia-Roberts,
Russell, Abate & Gleave (2023)**, which characterises the *partial identifiability
of reward functions* — multiple rewards fitting the same data — and asks when the
residual ambiguity is tolerable for a downstream task such as policy optimisation.
That is structurally our question with "reward function" replaced by "policy" and
"data source" by "evaluation protocol."

**What Collapse3 instantiates differently.** Partial-ID work is about *statistical*
ambiguity under sampling and modelling assumptions and typically yields *bounds
with inference*; ours is an *exact, finite, adversarial* identified set with no
sampling in the headline. Skalse et al. study which rewards are consistent with
observed (near-)optimal behaviour; we study which *policies* are consistent with
an evaluation's *observations* and how bad the worst is under a best-response
adversary. The mechanism ("optimal play refuses the lines that would disambiguate")
is a game-theoretic sharpening of their invariance story.

**Novelty verdict:** *new exact application / synthesis* — an identified set made
exact and adversarial inside a solved game. This is the relative we most owe a
citation, and the one least likely to have been connected to game evaluation.

## 4. Conformance & model-based testing

**Closest analogue.** Conformance testing asks whether an implementation's
observable behaviour is consistent with a specification — "consistency with what
was observed" is shared vocabulary.

**Established by.** Tretmans (1996) `ioco` (input-output conformance over labelled
transition systems); Lee & Yannakakis (1996) survey testing of finite-state
machines.

**What Collapse3 instantiates differently.** Conformance asks *does this one
implementation conform?*; we quantify *over all conforming implementations, how
bad is the worst outcome?* — a worst-case over the whole conforming class, graded
by an exact value, not a pass/fail on one artefact.

**Novelty verdict:** *new demonstration* (adjacent framing, different question).

## 5. Adversarial policies / adversarial evaluation

**Closest analogue.** The mechanism — a candidate that scores well is beaten by an
adversary playing *off the normal distribution* — is exactly the adversarial-policy
phenomenon, and Finding 17 makes it a property of the *evaluation protocol*.

**Established by.** Gleave et al. (2020) demonstrated adversarial policies against
self-play-robust RL agents; Wang et al. (2023) beat superhuman KataGo with a weak
adversary that "does not win by playing Go well… [but] tricks KataGo into serious
blunders" — the deployed-system version of this repo's Findings 14/16.

**What Collapse3 instantiates differently.** Those are *empirical attacks* on
specific trained agents; Finding 17 turns the same mechanism into an *exact
statement about which opponents an evaluator should use* (all-legal / suboptimal
probing), with the best-response adversary as an exhaustive certifier rather than
a learned attack.

**Novelty verdict:** *new exact application* (same phenomenon, stated as an
evaluation-design theorem in a solved game). Wang et al. is already cited in
[Finding 14](FINDINGS.md).

## 6. Formal verification / CEGAR

**Closest analogue.** Best-response with pinned nodes produces a
counterexample (a forcing line) that a "danger zone" was left uncovered — the
counterexample-as-diagnosis pattern of CEGAR.

**Established by.** Clarke, Grumberg, Jha, Lu & Veith (2000 CAV / 2003 JACM):
abstract, check, and refine the abstraction using spurious counterexamples.

**What Collapse3 instantiates differently.** CEGAR refines an *abstraction* until
a property is proved or refuted on the model; we hold the model exact and ask what
an *observation set* fails to certify about the true value. The counterexample is
a forcing line against a compatible policy, not a spurious abstract trace.

**Novelty verdict:** *new demonstration* (shared counterexample-driven spirit,
different target).

## 7. Policy / state equivalence in MDPs

**Closest analogue.** "Transcript-compatible policies" is an equivalence class of
policies indistinguishable under a given observation — an equivalence notion on
policies.

**Established by.** Givan, Dean & Greig (2003) formalise MDP state equivalence
(stochastic bisimulation) for model minimization; Ferns, Panangaden & Precup
(2004) give bisimulation *metrics*.

**What Collapse3 instantiates differently.** Bisimulation equivalences are defined
by *dynamics/optimal value* and used to *compress* the model; our equivalence is
defined by *what an evaluation observed* and used to *bound worst-case outcome*.
Different equivalence, different purpose.

**Novelty verdict:** *new demonstration*.

---

## Net placement

The method is **not** new: "the worst object still consistent with what a test
observed" is surviving-mutant reasoning (§1) and an identified set (§3), and the
attack mechanism is adversarial policies (§5). Collapse3's defensible
contribution is the **synthesis made exact**: computing that worst compatible
policy *exactly, against a solved game's ground truth*, and the resulting
evaluation-design result that **opponent strength ≠ evaluator strength** (the
exposure universe is the lever; strategic selection over it buys nothing — H2
falsified). Claim "new exact application / synthesis," cite §§1, 3, 5 prominently,
and do not assert a new method.

---

## References

- DeMillo, R. A., Lipton, R. J., & Sayward, F. G. (1978). Hints on Test Data
  Selection: Help for the Practicing Programmer. *Computer*, 11(4), 34–41.
- Jia, Y., & Harman, M. (2011). An Analysis and Survey of the Development of
  Mutation Testing. *IEEE Transactions on Software Engineering*, 37(5), 649–678.
- Harrold, M. J., Gupta, R., & Soffa, M. L. (1993). A Methodology for Controlling
  the Size of a Test Suite. *ACM TOSEM*, 2(3), 270–285.
- Manski, C. F. (1995). *Identification Problems in the Social Sciences*. Harvard
  University Press. — (2003). *Partial Identification of Probability
  Distributions*. Springer.
- Tamer, E. (2010). Partial Identification in Econometrics. *Annual Review of
  Economics*, 2, 167–195.
- Skalse, J., Farrugia-Roberts, M., Russell, S., Abate, A., & Gleave, A. (2023).
  Invariance in Policy Optimisation and Partial Identifiability in Reward
  Learning. *ICML* (PMLR 202:32033–32058).
- Tretmans, J. (1996). Test Generation with Inputs, Outputs and Repetitive
  Quiescence. *Software—Concepts and Tools*, 17(3), 103–120.
- Lee, D., & Yannakakis, M. (1996). Principles and Methods of Testing Finite-State
  Machines — A Survey. *Proceedings of the IEEE*, 84(8), 1090–1123.
- Gleave, A., Dennis, M., Wild, C., Kant, N., Levine, S., & Russell, S. (2020).
  Adversarial Policies: Attacking Deep Reinforcement Learning. *ICLR*.
- Wang, T. T., et al. (2023). Adversarial Policies Beat Superhuman Go AIs. *ICML*
  (PMLR 202:35655–35739).
- Clarke, E., Grumberg, O., Jha, S., Lu, Y., & Veith, H. (2000/2003).
  Counterexample-Guided Abstraction Refinement. *CAV* (LNCS 1855) / *JACM*,
  50(5), 752–794.
- Givan, R., Dean, T., & Greig, M. (2003). Equivalence Notions and Model
  Minimization in Markov Decision Processes. *Artificial Intelligence*,
  147(1–2), 163–223.
- Ferns, N., Panangaden, P., & Precup, D. (2004). Metrics for Finite Markov
  Decision Processes. *UAI*, 162–169.
