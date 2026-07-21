# Collapse3 Explained in 30 Seconds

Board notation: see [`NOTATION.md`](NOTATION.md). **O = you, X = opponent.**
Each cell is a peg; a bead falls to the lowest empty level. The three panes are
the grid rows **Front / Middle / Back**; inside a pane the lines run Level 3
(top) → Level 1 (floor). *Every board below is produced by the engine and
verified in `tests/test_thirty_seconds.py`.*

---

**Get three of your beads in a row.**

## Winning positions

**1. Flat — three on one level**

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ O │ O │ O │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

**2. Vertical — three on one peg**

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ O │ - │ - │   Level 3 (top)
      │ O │ - │ - │   Level 2
      │ O │ - │ - │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

**3. Staircase — climbing one level per step**

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ O │   Level 3 (top)
      │ - │ O │ X │   Level 2
      │ O │ X │ X │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

---

## The Gravity Collapse

Instead of placing, you may **remove** an opponent bead. It is destroyed, and
everything above it **falls**.

**1. How it works** — remove the Middle-centre bottom X; the beads above drop one
level.

BEFORE:

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ - │ - │ - │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ X │ - │
      │ - │ O │ - │
      │ - │ X │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

AFTER:

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ - │ - │ - │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ X │ - │
      │ - │ O │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

**2. "Oops"** — if your collapse drops the *opponent* into a row, **they win.**
Here you remove the Front-left bottom bead and the X's slide into a line.

BEFORE:

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ X │ - │   Level 3 (top)
      │ X │ O │ X │   Level 2
      │ O │ X │ X │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

AFTER (X wins):

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ X │ X │ X │   Level 2
      │ O │ O │ X │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

**3. The clever collapse** — a removal can **win on the spot.** Remove the
Front-centre bottom X; your O rides gravity down into the winning row.

BEFORE:

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ X │ - │   Level 3 (top)
      │ - │ O │ - │   Level 2
      │ O │ X │ O │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

AFTER (O wins):

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ X │ - │   Level 2
      │ O │ O │ O │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ - │ - │ - │
      │ - │ - │ - │
      └───┴───┴───┘
```

---

**One rule to remember: cooldown.** You cannot collapse twice in a row — after a
removal, your next turn must be a placement (if one is legal).

*Full game: [`rules.md`](rules.md) · why it's more than it looks:
[`NUTSHELL.md`](NUTSHELL.md).*
