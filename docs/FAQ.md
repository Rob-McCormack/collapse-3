# FAQ

Short answers to recurring questions. Detail and numbers live in
[`FINDINGS.md`](FINDINGS.md); the rules live in [`../rules.md`](../rules.md).

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
  Trained agents may still pay almost none of that floor on-policy if they steer
  around aliased states — the floor is uniform-over-states, play is not.

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

Exact enumeration tops out around (5,5) (~12.7M states); the full game needs an
**approximate oracle** (e.g. strong MCTS) with bounded error, or a
solver-generated curriculum of critical states. The exact curve motivates the
hypothesis that these effects are structural; testing it at full scale is future
work. See "Limitations & scaling" in [`FINDINGS.md`](FINDINGS.md).

## 8. What's the "superposition" framing about — isn't that overreach?

It's an optional *interpretation*, kept deliberately separate from the empirical
claims. The idea is that each mechanic forces a bounded agent to act on
**unresolved information**, and the exact solver is what resolves it. None of it
is quantum — it's a framing device, not a claim. If it helps, read
[`PERSPECTIVE.md`](PERSPECTIVE.md); if it doesn't, ignore it — the results in
[`FINDINGS.md`](FINDINGS.md) stand on their own.

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
perfect player** (1825 vs 1804): the equal game is drawn, so "unbeatable" caps
what optimal play can earn, and the surplus rating comes from converting wins
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
