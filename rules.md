# Rules

# Collapse3
> Version 1.0.0 July 6, 2026

**A 3D spatial strategy game of gravity, structural shielding, and shifting foundations.**

## Objective

Be the first player to align three of your beads in a straight line on the 3×3×3 board.

A line may be vertical, horizontal, flat diagonal, or staircase diagonal.

If no player achieves a 3-in-a-row before the game reaches a tiebreaker condition, the player with the most surviving beads on the board wins.

---

## Components

* 1 base with a 3×3 grid of pegs.
* Each peg holds up to 3 beads.
* 28 beads total: 14 beads per player.

---

## Setup

1. Clear the board.
2. Each player takes 14 beads. These are their **Reserve**.
3. Decide who goes first.

---

## Board Structure

The board has 9 pegs arranged in a 3×3 grid.

Each peg has 3 possible bead levels:

* **Level 1:** bottom
* **Level 2:** middle
* **Level 3:** top

When a bead is placed on a peg, it always falls to the lowest empty level on that peg.

---

## Turn Structure

Players take turns. On your turn, you must perform exactly one legal action:

1. **Placement**, or
2. **Removal**, also called **The Collapse**.

If Removal is not legal and Placement is legal, you must make a Placement.

If neither Placement nor Removal is legal, the game immediately ends by surviving-bead count.

---

## Action 1: Placement

To place a bead, choose any peg that is not full and slide one bead from your Reserve onto that peg.

The bead falls to the lowest empty level.

A Placement is legal only if:

* you have at least one bead in your Reserve, and
* at least one peg has an empty space.

After placing, check immediately for a 3-in-a-row.

---

## Action 2: Removal — The Collapse

Instead of placing, you may remove one opponent bead from the board, but only if all Removal conditions are satisfied.

The removed bead is permanently destroyed. It does **not** return to either player’s Reserve.

After the bead is removed, gravity applies immediately: any beads above the removed bead on that peg drop down to fill the gap.

After the gravity cascade resolves, check immediately for a 3-in-a-row.

---

## Conditions for a Legal Removal

You may perform a Removal only if all of the following conditions are met:

### 1. Cooldown

You cannot perform a Removal on two of your consecutive turns.

If you removed a bead on your previous turn, your next turn must be a Placement if Placement is legal.

### 2. Singleton Immunity

You cannot remove from a peg that contains only 1 bead.

The targeted peg must contain at least 2 beads.

### 3. Tallest Stack

You may only remove from a peg that is tied for the tallest stack on the board.

For example, if the tallest stacks are height 3, you may only remove from a height-3 peg. If the tallest stacks are height 2, you may only remove from a height-2 peg.

### 4. Capping Rule

The top bead of the targeted peg must belong to your opponent.

If your own bead is on top of a tallest stack, that stack is protected from your Removal action.

### 5. Target Rule

If a peg is valid for Removal, you may remove any opponent bead from that peg, including a bottom, middle, or top bead.

You may not remove your own bead.

---

## Winning by 3-in-a-Row

The game ends immediately when a player forms three of their beads in a straight line.

There are three types of winning lines.

### 1. Vertical

Three beads of the same player stacked on one peg.

### 2. Flat Line

Three beads of the same player on the same level across a straight row, column, or diagonal of the 3×3 grid.

Flat lines may occur on Level 1, Level 2, or Level 3.

### 3. Staircase Line

Three beads of the same player on three collinear pegs of the 3×3 grid, with levels ascending or descending one step at a time.

Valid staircase patterns are:

* Level 1 → Level 2 → Level 3
* Level 3 → Level 2 → Level 1

The three pegs must form a straight row, column, or diagonal on the base grid.

---

## Accidental Wins: The Oops Rule

If your Removal causes a gravity cascade that creates a 3-in-a-row for your opponent, your opponent wins immediately.

This can happen even though you were the active player.

---

## Simultaneous Wins

If a single action results in both players having a 3-in-a-row at the same time, the active player wins the tie.

The active player is the player whose turn caused the simultaneous result.

---

## Exhaustion and No-Legal-Move Tiebreaker

The game ends by surviving-bead count if either:

* both players’ Reserves are empty and no player has achieved a 3-in-a-row, or
* the active player has no legal action.

When this happens, count the total number of each player’s beads surviving on the board.

The player with more surviving beads wins.

If both players have the same number of surviving beads, the game is a true Draw.

---

## Reserve and Destruction Clarifications

A bead in Reserve has not yet entered the game.

A bead on the board is surviving.

A bead removed by The Collapse is permanently destroyed.

Destroyed beads are not counted as surviving beads and do not return to Reserve.

---

## End of Game Priority

After every action, resolve the game in this order:

1. Complete the Placement or Removal.
2. If Removal occurred, resolve gravity.
3. Check for 3-in-a-row.
4. Apply the Oops Rule if the opponent alone has won because of your Removal.
5. Apply the Simultaneous Wins rule if both players have a 3-in-a-row.
6. If no 3-in-a-row exists, check whether a tiebreaker condition has been reached.
7. If no win or tiebreaker condition applies, play passes to the next player.

