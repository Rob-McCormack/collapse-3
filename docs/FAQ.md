# FAQ

Short answers to recurring questions. Detail and numbers live in
[`FINDINGS.md`](FINDINGS.md); the rules live in [`../rules.md`](../rules.md).

## Questions at a glance

1. **[Is Collapse3 a POMDP?](#1-is-collapse3-a-pomdp--partially-observable)** — the game is perfect-information; the *experiments* induce a POMDP by restricting observation.
2. **[Isn't the current state always sufficient?](#2-isnt-the-current-state-always-sufficient--like-in-chess)** — yes for the full state; no for a lossy view of it.
3. **[Is hiding reserves "cheating"?](#3-is-hiding-reserves-cheating-in-a-perfect-information-game)** — no: it is an ablation that prices a missing feature exactly.
4. **[Is the (6,6) first-player win a bug?](#4-is-the-66-first-player-win-a-solver-bug)** — no; equal reserves become a P0 win from (6,6) through (14,14).
5. **[Won't more data / bigger models fix this?](#5-wont-more-data-bigger-models-or-better-training-fix-this)** — not the representation floors: the missing information is absent from the input.
6. **[Why value-based regret instead of win rate?](#6-why-value-based-regret-instead-of-win-rate)** — win rate convolves agent and opponent; regret grades each decision against the oracle.
7. **[Does any of this scale to the real game?](#7-does-any-of-this-scale-to-the-real-game-14-beads)** — yes: (14,14) is solved, and the full-size battery grades every move.
8. **[Don't the findings just restate "PO is hard"?](#8-dont-the-findings-just-restate-partial-observability-is-hard)** — yes, with *exact* per-interface prices and scoreboard unit-tests.
9. **[Someone proposed a simple strategy — what happened?](#9-someone-proposed-a-simple-strategy--what-happened)** — certified draw at (3,3); certified forced loss from (4,4) up.
10. **[Why not just rank agents by Elo?](#10-why-not-just-rank-agents-by-elo)** — Elo ranked an exploitable rulebook above the perfect player (+21.1; CI excludes 0).
11. **[Could a grandfather let his grandson win every time?](#11-could-a-grandfather-let-his-grandson-win-every-time)** — not by force; the oracle audits thrown value, not intent.
12. **[You solved it — so what's the strategy?](#12-you-solved-the-game--so-whats-the-strategy-is-the-theoretical-solution-meaningful)** — computationally solved ≠ strategically compressible.
13. **[Why score wins 100 / 10 / 0 / −10 / −100?](#13-why-does-the-solver-score-wins-100--10--0----10----100)** — ordinal for move choice; headline metrics use magnitude-free WDL units.
14. **[Does this mean AI is "hitting a wall"?](#14-does-this-mean-ai-is-hitting-a-wall)** — no: the urgent split is capability vs. what we can still certify.
15. **[If a learner plays it perfectly, why "calculable but not teachable"?](#15-if-a-learner-can-play-it-perfectly-why-call-it-calculable-but-not-teachable)** — learnable by memorizing/searching; not reducible to a compact transferable rulebook.
16. **[Can the agent decide *when* to gather more information?](#16-can-the-agent-decide-when-to-gather-more-information)** — yes, exactly: missing ≠ useful, and from (4,4) up only ~1% of decisions must inspect reserves, yet that 1% carries the whole floor.

---

## 1. Is Collapse3 a POMDP / partially observable?

The *game* is a perfect-information, deterministic MDP — nothing is hidden from
anyone: buried beads, reserves, and cooldown are all on the table, and the full
state is Markov-sufficient. The *experiments* construct a POMDP from it by
restricting what the agent observes — with the unusual property that the
underlying MDP is exactly solved, so the cost of partial observability can be
**enumerated rather than bounded**. That cost is the irreducible floor for a
*memoryless* policy; a history/memory-conditioned agent recovers ground truth
(Finding 7).

Concretely: we feed a *memoryless* policy a **lossy observation** of the state
(e.g. reserves hidden), so two genuinely different states collapse to one
observation — classic perceptual (state) aliasing, the "memoryless policy in a
POMDP" special case.

The difference from a real POMDP is the point of the project:

- **We have ground truth.** The underlying MDP is fully solved, so we don't
  *estimate* the cost of partial observability — we compute the **exact
  irreducible regret floor** of any memoryless policy over that observation
  (Finding 4). A genuine POMDP rarely gives you that.
- **The POMDP resolution appears cleanly.** A history/memory-conditioned policy
  can reconstruct the dropped fields from observed removals — exactly the
  belief-state answer POMDP theory predicts. At (4,4) the memory-augmented floor
  is **0.0000** by enumeration (Finding 7); the memoryless floor is **0.0805**.
  Trained agents may still pay almost none of that floor on-policy — their real
  interface (the legal-move mask) already erases most of it, and their trajectories
  steer around the rest; the floor is uniform-over-states, play is not.

One-liner: **a solved perfect-information game, with aliasing induced on purpose
to price it exactly — the opposite of an intractable POMDP.**

## 2. Isn't the current state always sufficient — like in chess?

The *full* state is always sufficient (both games are Markov). The point is that
**what you observe is not always the state.** In chess the position on the board
essentially *is* the full state, so learning straight from it is fine. In
Collapse3, what you'd read off the top — the visible tops — is *not* the state:
buried beads, reserves, and cooldown decide outcomes but aren't visible up top.
The honest slogan is "the obvious observation isn't the state," and Collapse3
lets us put an exact price on that gap.

## 3. Is hiding reserves "cheating" in a perfect-information game?

No. We're not adding secrecy to the game; we're restricting the **agent's input**
and measuring what that restriction costs. The cost is **aliasing** (two states
sharing one observation), not hidden information — see the "Aliasing, not
secrecy" note in [`FINDINGS.md`](FINDINGS.md). It models the everyday reality
that a learned system rarely gets the complete state as input.

## 4. Is the (6,6) first-player win a solver bug?

We don't think so. Exact enumeration is infeasible at (6,6), so it's proven by
the folded solver alone — but that solver is hardened four ways:

1. **Cross-checked against an independent enumeration solver on every size where
   both can run** (see `experiments/opening_values.py` and
   `tests/test_opening_values.py`); they agree everywhere.
2. **Order-invariance.** Alpha-beta + transposition-table bugs are notoriously
   sensitive to move-exploration order, so we re-solve (6,6) from a *fresh*
   table under five shuffled move orderings — the value is 100 every time.
3. **Principal-variation replay.** We extract an optimal line from the empty
   (6,6) board (9 positions) and independently re-verify each position's value
   from a fresh, shuffled table.
4. **Pattern consistency.** "Mover never loses" holds across the *whole* grid,
   not just the diagonal; a localized bug would likely break the pattern
   off-diagonal before it flipped a single cell.

The result is also game-theoretically sensible: with enough material the mover
can build threats faster than the cooldown-limited defender can remove them.
See Finding 6.

## 5. Won't more data, bigger models, or better training fix this?

**Does this deny that scaling works?** No — it shows scaling is *conditional*:
more data and compute fix a weak training signal, but cannot cross a
representation floor (enumerated, Finding 4) or repair an evaluation that
structurally can't see the failure (Findings 8, 10, 14).

Only if the failure is in the *objective*, not the *representation*. The same
full-state learner trained against a random opponent wins ~98% but carries real
regret (0.0581); trained against an optimal opponent the regret collapses ~30×
(0.0019) — **objective** failure, fixable by a stronger signal (Finding 5). But
feed the agent a **lossy** observation (reserves hidden) and the exact aliasing
floor *grows* with size — 0.0028 at (3,3), 0.0805 at (4,4), 0.1677 at (5,5) —
proven by enumeration, not by training curves (Finding 4; a budget sweep can
only show scaling converging to whatever the agent's own interface floor is —
see the correction note in "Is this just undertraining?"). If two states
collapse to one
observation but need different moves, any memoryless policy *on that interface*
is wrong on at least one, regardless of capacity. That is **representation**
failure — a wall scaling cannot break. One honest caveat: the wall is a property
of the *interface*, not of training. Widen the interface — memory (Finding 7),
or even just the legal-move list, which leaks hidden state — and the floor drops
or vanishes; each interface's floor is priced exactly in Finding 4. Full
argument: "Is this just undertraining?" in [`FINDINGS.md`](FINDINGS.md).

## 6. Why value-based regret instead of win rate?

Win rate convolves the agent's skill with the opponent's policy *and* the seat it
played — it can't tell a strong agent from one that just faced weak opponents, or
moved first. Value-based regret grades each decision against the exact oracle: it
is per-decision, opponent-independent, and seat-independent. We use *value*
regret rather than policy distance because optimal play here is highly non-unique
(many moves hold a draw), so policy distance would wrongly penalize
correct-but-different moves.

## 7. Does any of this scale to the real game (14 beads)?

Two-part answer — and the second half changed when we actually tried.

**The root is solved.** The full (14,14) opening is a **first-player forced true win**
(+100), certified by the repo's exact solver in ~96K nodes (~3 seconds). The
14×14 opening grid extends the published 6×6 table without changing a cell;
diagonal (6,6)–(14,14) are all first-player wins. Verified independently by the
clean-room reference engine and by a raw alpha-beta with no transposition table
and no symmetry folding (42M nodes). See Finding 11 in [`FINDINGS.md`](FINDINGS.md).

**Enumeration still tops out near (5,5).** Solving the root is not visiting
every reachable state. Aliasing floors, distribution-free census, and misère
solves still require full enumeration and remain capped at the smaller sizes.

**Every real game is exactly gradeable.** Once the root is cheap, a node-capped
solver grades **100% of moves** in every full-size game we played across the
agent zoo (`experiments/horizon.py`). The "approximate oracle" for play
evaluation is no longer future work — optimal play at (14,14) costs seconds per
move.

## 8. Don't the findings just restate "partial observability is hard"?

Yes — and with exact numbers. The classical names for what Collapse3 measures
are already the right ones: **aliasing**, **partial observability**, and
**bounded rationality**. The contribution is not a new vocabulary; it is that
the cost of a missing feature becomes an *enumerated* floor per interface, that
the floor's shape can be mapped across sizes on sibling boards, and that
standard scoreboards (win rate, Elo, on-policy regret) can be unit-tested
against those floors. See [`FINDINGS.md`](FINDINGS.md).

## 9. Someone proposed a simple strategy — what happened?

An external model (Kimi) proposed a five-line rulebook: take a win if you have
one, otherwise place centre → corners → edges, block threats, remove only to
win. We froze it as a named agent and solved the game *against it* exactly
(`experiments/best_response.py`): one formalization is a **certified exact
drawing policy at (3,3)** — both seats, all opponents — and **every
formalization is a certified forced loss from (4,4) up**, with the kill just
5–8 plies deep. The same policy had survived 1,199 of 1,200 games against a
strong noisy opponent, which is the real lesson: game-playing evaluation missed
a 5-ply refutation that the exact solve finds in a fraction of a second. Anyone
can submit a deterministic policy and get a certificate or a refutation — see
Finding 8 in [`FINDINGS.md`](FINDINGS.md).

## 10. Why not just rank agents by Elo?

Because we ran that tournament, and Elo got the top of the table wrong. In a
nine-agent round-robin at (4,4) — 3,600 oracle-graded games, ratings fitted by
maximum likelihood — **Elo ranks a certifiably exploitable rulebook above the
perfect player** (1825 vs 1804, **+21.1**; bootstrap 95% CI **[5.1, 38.3]**,
P(gap > 0) = 0.996): the equal game is drawn, so "unbeatable" caps what
optimal play can earn, and the surplus rating comes from converting wins
against weak opponents — a skill the value-preserving optimal agent has no
reason to have. The truthful columns from the *same games* rank differently:
the optimal agent is the only one with zero losses and zero regret, and its
worst case is a theorem while the table-topper's is a certified 7-ply forced
loss. Elo's worst single prediction: a 286-point gap that implies an 84%
expected score lands on a pure first-mover seat split (50/50). Full table and
mechanism in [`FINDINGS.md`](FINDINGS.md) (Finding 10); see also #6.

## 11. Could a grandfather let his grandson win every time?

Not by force — and the game is oddly stubborn about it (call it *the Grandfather
Problem*). We solved it exactly: a player who *wants* to lose **cannot force the
opponent to win** at any solved size ((3,3)–(5,5)), from either seat. The reason
is that the game's automatic win conditions turn against the thrower — his only
removals delete the opponent's beads (tipping the surviving-bead tiebreaker back
toward *him*), and the opponent can bounce his own placements into accidental
wins via gravity. He *can* guarantee he never wins (accept a draw), and against a
random opponent he can make them win more and more often as the game grows
(≈31% → 80% at (5,5) going first) — but never 100%. For evaluations the lesson
is two-sided: actually *giving away* the game always logs positive regret with
the oracle (a thrown move scores like a genuine blunder), but grandpa can
freely pick **among equally-optimal moves** the one his grandson is most likely
to win from — and that choice is invisible to any value audit. The oracle sees
thrown value, not thrown intent. Details in [`FINDINGS.md`](FINDINGS.md)
(Finding 9).

## 12. You solved the game — so what's the strategy? Is the theoretical solution meaningful?

Traditionally, a "theoretical solution" means more than knowing the game's value
or the correct move in every position. It implies a coherent, reasonably compact
theory of play: principles that explain what matters and how correct decisions
can be derived without calculating every state.

Collapse3 creates tension with that idea. Because it is finite and deterministic,
an exact solver can determine every position's value and generate a complete
contingent policy. But that may amount only to an enormous collection of
state-specific instructions — a highly detailed roadmap **without a legend**. It
tells you where to turn at every junction; it may not explain the landscape.

The research suggests the game can be **computationally solved without being
strategically compressible**. Compact rulebooks succeed at the smallest size and
receive shallow forced refutations as it grows ([Finding 8](FINDINGS.md)).
Apparently natural principles reverse with scale: the centre opening can go from
winning to drawing to losing ([Findings 12–13](FINDINGS.md)), and intuitive
line-building can perform worse than random. Correct action increasingly depends
on detailed calculation — reserves, buried beads, cooldown, gravity, and the
opponent's replies — not on a teachable rule.

Self-play shows the same gap. An agent can learn a certified winning corridor
from the start without acquiring a general theory of the game or playing
correctly over the wider state space ([Finding 14](FINDINGS.md)). It has learned
how to navigate one robust route, not *why* the game works.

So the important distinction is between an **exact computational solution** and
a **traditional theoretical solution**. The first clearly exists — it ships in
this repository. The second may not exist in any useful, compact, or
human-interpretable form. Collapse3 may be completely mapped while remaining
strategically unexplained — and that tension is central to the research.

## 13. Why does the solver score wins 100 / 10 / 0 / −10 / −100?

These encode an **ordering** — true win > attrition win > draw > attrition loss
> true loss — and for **move selection they are purely ordinal**. The solver
only ever compares values (`min`/`max`, never sums), so relabelling them
2 / 1 / 0 / −1 / −2 would pick the same moves, label the same moves optimal,
give the same optimal-move rates, and certify the same Gate C outcomes. The
game-theoretic winner is untouched.

The magnitudes matter in exactly **one** place: the *weighted* regret metric,
which is the drop in raw score (`best_mover_value − mover_value`). There a
move that turns a draw into a true loss logs **100** while a draw into an
attrition loss logs **10** — a deliberate 10× weighting, not an economic claim
that a regulation win is "worth" ten attrition wins. Because that scale is a
modelling choice, the **headline competence numbers are reported in
magnitude-free units**: **WDL regret** (0/1/2) and the ordinal **`class_regret`
ladder** (steps between outcome classes). The weighted score is a secondary
lens.

This is also why regret figures differ by orders of magnitude between findings:
a mean like **21–64** is weighted solver-units (Finding 1), while **0.002–0.25**
is per-decision WDL units averaged over all decisions (Findings 5, 7). Same
underlying labels, two scales — always check which one a table names.

## 14. Does this mean AI is "hitting a wall"?

No — and the distinction matters.

Collapse3 shows that **exact reasoning can punch through walls that brute-force
enumeration cannot**. The (14,14) root solves in ~96K nodes despite a state
space far beyond enumeration ([Finding 11](FINDINGS.md)). The "wall" is not at
the frontier of capability.

The wall is at the frontier of **what we can certify**. We can solve the root;
we cannot enumerate the aliasing floors, the census, or the misère properties
at full size. We can grade specific moves; we cannot verify global behavior
without visiting every state. We can train an agent to look perfect on its own
trajectories; we cannot prove it is robust off them ([Finding 14](FINDINGS.md)).

In real AI systems, the same split applies: models may keep getting more capable
while our ability to evaluate them scales worse. The "hitting a wall" discourse
usually asks "can AI keep getting better?" Collapse3 suggests the more urgent
question is "can we keep knowing whether it's better?"

This is why the repo focuses on **evaluation methodology**, not model design.
Some of the failure modes measured here — win-rate deception, Elo inversion,
invisible steering — are pathologies of the *evaluator*. Others — aliasing
floors, interface leakage, coverage decay — are priced properties of the
*agent's interface* or training distribution. Fixing either class requires
better measurement, not just more compute.

## 15. If a learner can play it perfectly, why call it "calculable but not teachable"?

Both are true, and the tension is the point. The repo's own tabular Q-learner
(full state, no oracle access) reaches **0.0019** mean regret and **99.8%**
optimal moves — essentially solver-perfect ([Finding 2](FINDINGS.md)). So the
game is absolutely *learnable*.

"Not teachable" is a different claim: there is no compact, shallow rulebook you
could write down and hand to a beginner. A naive "build your own lines"
heuristic is optimal on **0%** of critical decisions; a threat-aware one-ply
rulebook still loses **64%** of a drawn game; competence appears only at 2-ply
search (**85%**) and perfection at 6-ply (**100%**) ([Finding 8](FINDINGS.md)).
The agent succeeds by **memorizing a large table** (~22k–28k states per seat at
(4,4)) or by **searching** — not by distilling a generalizable principle. Solver
and learner reach the same competence in different forms; **neither discovers a
compressible, human-interpretable strategy** (see also #12).

Why it matters — and how it differs from the representation floor of
[Finding 4](FINDINGS.md): here the state is **fully observed**, so scaling the
learner works, but only via lookup (needs a small state space to tabulate) or
search (needs the search to be affordable). That is a *separate* failure from
the aliasing floor, which comes from a **lossy** observation, not from the
difficulty of generalizing. If a real task shares this structure — correct
actions set by long, non-local causal chains that don't project onto local
features — then near-zero regret on one opponent distribution need not transfer
to another (Finding 2), and there is no shortcut around search or coverage.

## 16. Can the agent decide *when* to gather more information?

Yes — and because the game is solved, we can grade that decision exactly. Instead
of asking "what does a fixed missing feature cost?" ([Finding 4](FINDINGS.md)),
[Finding 15](FINDINGS.md) asks the adaptive question a real agent faces: given a
cheap view, *when should it decide its view is insufficient and go acquire more
before acting?* For each observation group we compare acting now against querying
a richer view; the per-group optimum is an exact **information policy**.

Two results stand out. First, **missing is not the same as useful**: at (3,3)
reserves are aliased away in 36% of decision states, yet inspecting them buys
**zero** — the optimal number of reserve queries is zero. Second, from (4,4) up
the value of information turns on, but it is **sparse**: only **~1.07%** of
decisions need reserves, and that thin slice carries the *entire* irreducible
floor — querying there drives it to exactly 0. (The legal-action mask, by
contrast, is *diffusely* useful — ~13% of states — but it normally ships for
free, so it's only an illustration.)

This is the exact, game-based version of *value of information* (Howard 1966) and
metareasoning about when to sense/compute (Russell & Wefald 1991), and it maps
directly onto modern agents deciding when to retrieve a document, call a tool,
consult memory, or think longer before answering. Numbers guarded by
`tests/test_info_policy.py`; see `results/info_policy_latest.json`.
