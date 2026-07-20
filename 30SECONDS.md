# Collapse3 Explained in 30 Seconds

Board notation: see [`NOTATION.md`](NOTATION.md). **O = you, X = opponent.**
Each cell is a peg; a bead falls to the lowest empty level. *Every board below is
produced by the engine and verified in `tests/test_thirty_seconds.py`.*

---

**Get three of your beads in a row.**

## Winning positions

**1. Flat — three on one level**

```text
┌───┬───┬───┐
│ - │ - │ - │
│ - │ - │ - │
│ O │ O │ O │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
└───┴───┴───┘
```

**2. Vertical — three on one peg**

```text
┌───┬───┬───┐
│ O │ - │ - │
│ O │ - │ - │
│ O │ - │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
└───┴───┴───┘
```

**3. Staircase — climbing one level per step**

```text
┌───┬───┬───┐
│ - │ - │ O │
│ - │ O │ X │
│ O │ X │ X │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
├───┼───┼───┤
│ - │ - │ - │
│ - │ - │ - │
│ - │ - │ - │
└───┴───┴───┘
```

---

## The Gravity Collapse

Instead of placing, you may **remove** an opponent bead. It is destroyed, and
everything above it **falls**.

**1. How it works** — remove the bottom X; the beads above drop one level.

```text
        BEFORE                    AFTER
┌───┬───┬───┐            ┌───┬───┬───┐
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
├───┼───┼───┤            ├───┼───┼───┤
│ - │ X │ - │            │ - │ - │ - │
│ - │ O │ - │            │ - │ X │ - │
│ - │ X │ - │            │ - │ O │ - │
├───┼───┼───┤            ├───┼───┼───┤
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
└───┴───┴───┘            └───┴───┴───┘
```

**2. "Oops"** — if your collapse drops the *opponent* into a row, **they win.**
Here you remove the centre's bottom bead and the X's slide into a line.

```text
        BEFORE                    AFTER  (X wins)
┌───┬───┬───┐            ┌───┬───┬───┐
│ - │ X │ - │            │ - │ - │ - │
│ X │ O │ X │            │ X │ X │ X │
│ O │ X │ X │            │ O │ O │ X │
├───┼───┼───┤            ├───┼───┼───┤
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
├───┼───┼───┤            ├───┼───┼───┤
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
└───┴───┴───┘            └───┴───┴───┘
```

**3. The clever collapse** — a removal can **win on the spot.** Remove the
centre's bottom X; your O rides gravity down into the winning row.

```text
        BEFORE                    AFTER  (O wins)
┌───┬───┬───┐            ┌───┬───┬───┐
│ - │ X │ - │            │ - │ - │ - │
│ - │ O │ - │            │ - │ X │ - │
│ O │ X │ O │            │ O │ O │ O │
├───┼───┼───┤            ├───┼───┼───┤
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
├───┼───┼───┤            ├───┼───┼───┤
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
│ - │ - │ - │            │ - │ - │ - │
└───┴───┴───┘            └───┴───┴───┘
```

---

**One rule to remember: cooldown.** You cannot collapse twice in a row — after a
removal, your next turn must be a placement (if one is legal).

*Full game: [`rules.md`](rules.md) · why it's more than it looks:
[`NUTSHELL.md`](NUTSHELL.md).*
