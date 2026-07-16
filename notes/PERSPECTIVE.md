# Superposition: a conceptual framing

> **Not part of the audited research.** This note was moved out of `docs/` so a
> reviewer auditing the repo encounters only falsifiable claims. The phenomena it
> gestures at already have standard names used correctly in the findings —
> aliasing, partial observability, bounded rationality — and the measurements
> live in [`../docs/FINDINGS.md`](../docs/FINDINGS.md). Kept here only as a
> personal framing note; nothing below is a result.
>
> Nothing here is quantum — "superposition" is a metaphor for information that
> stays *unresolved* at the interface between a bounded agent and the game. If
> the framing doesn't help you, skip it.

Collapse3 rests on four kinds of unresolved information. None are physical. But
each one forces a bounded agent to act before the relevant fact is settled — and
that is exactly why an exact solver is useful: it settles the fact, so we can
measure what the agent's uncertainty cost.

**Observational aliasing.** Feed an agent a lossy observation and distinct states
collapse into the same input. The agent is stuck choosing one action for two
realities, and it pays for the ambiguity. The exact regret floor grows with the
game — **0.0805 at (4,4), 0.1677 at (5,5)** — and no training budget closes it,
because the distinction simply isn't in the input (Finding 4). This is the one
that is literally about the agent's *information*.

**Interaction, not ownership.** Removal hits your opponent's material, not your
own, and gravity redistributes the board afterward. A static plan assumes you own
your geometry; the opponent's move breaks that assumption. Re-solving each turn
isn't extra sophistication — it just acknowledges that the other agent has
already changed the state you were planning over. Adapt, or lose to strictly
worse play (Finding 3).

**Competing win conditions.** Two terminal conditions — a true (line) win and an
attrition win — stay live until the last move. Ranking them on an **ordinal
ladder** (true win > attrition win > draw > …) turns "who won" into a graded
signal. Binary win/loss throws away exactly the gradation that separates
competent play from lucky play — which is why competence here is *value-based*
regret, not a win rate.

**Locally brilliant, globally fatal.** A removal can look great up close —
destroying opponent material — while being globally losing, triggering an "Oops"
cascade that hands over the game. To a bounded agent the move's value is
unresolved: promising and suicidal at once, until search collapses the branch to
a single number. Those traps are rare but real (structural census).

These aren't physical superpositions; they're **computational epistemologies** —
ways information stays unresolved between agent and environment. The solver
doesn't observe wavefunctions. It brute-forces the tree and returns one value
where a bounded agent sees only possibility.

> The superposition isn't on the board. It's in the mind of the agent.

And that is the whole thesis in one line: the board is fully determined
(perfect-information, deterministic, solved). Any "indeterminacy" lives in the
agent's information about it — which is precisely why competence is a property of
the agent-plus-observation, not of the world.

## A note on causality

The mechanics encode a kind of **non-local** causality. A removal at one peg can,
via the gravity cascade, complete a line on the *opposite* side of the board. An
early placement can sit inert for many turns until a later removal drops it into a
winning position. The Oops rule is the extreme case: your own aggressive move
causes your defeat, through physics you triggered.

A bounded agent sees only the local move; the exact solver traces the full chain
to its terminal consequence. So value-based regret is a **counterfactual**
quantity — the value you would have kept under optimal continuation minus what
your move actually yields — not a mere correlation between action and outcome.
(Deliberately *counterfactual*, not "causal" in the interventionist sense: the
game has no interventions or confounders, just a fully determined tree the solver
reads to the end.) This is interpretive framing; the measured quantities remain
the regret floors and the inversion counts in [`FINDINGS.md`](FINDINGS.md).
