# Collapse3 Board Notation

To visualize, share, and debug game states in plain text, Collapse3 uses a
vertically-stacked ASCII notation that mirrors how beads physically stack on the
3×3 grid of pegs. It is produced by `collapse3/render.py`
(`render_board` / `render_state`), and the examples below are locked
byte-for-byte to that code by `tests/test_render.py`, so this document can never
drift from what the engine actually prints.

Conventions: **O** = player 0, **X** = player 1, and a hyphen (`-`) is an empty
slot. The board is one 3×3 grid of pegs, drawn as three stacked panes — the grid
rows **Front / Middle / Back**; the three columns of each pane are the
**left / center / right** pegs. Inside a pane the three lines are the three
levels, with **Level 3 on top and Level 1 (the floor) at the bottom** — so
gravity reads correctly and you never see a bead floating above a `-`.

## The blank board

Labels are just orientation aids — the panes are grid rows (not heights), and the
vertical axis inside every pane runs Level 3 (top) → Level 1 (floor):

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ - │ - │ - │   Level 1 (floor)
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

## Reading a position

A bead always falls to the lowest empty level, and the notation mirrors that:
beads fill each cell from the bottom up. Here O has completed a **flat
horizontal line** across the **Middle** row of the grid at Level 1:

```text
      ┌───┬───┬───┐   columns: left · center · right
Front │ - │ - │ - │   Level 3 (top)
      │ - │ - │ - │   Level 2
      │ - │ X │ - │   Level 1 (floor)
      ├───┼───┼───┤
Mid   │ - │ - │ - │
      │ - │ X │ - │
      │ O │ O │ O │
      ├───┼───┼───┤
Back  │ - │ - │ - │
      │ X │ - │ - │
      │ X │ - │ - │
      └───┴───┴───┘
```

How to read it:

- **Gravity is visible.** Beads stack from the bottom of each cell upward, so a
  `-` never appears *under* a bead in a legal position.
- **The winning line.** The **Middle** pane shows three `O`s spanning
  left to right at Level 1 — a flat three-in-a-row.
- **Stacking (centre peg).** The Middle pane's centre peg holds two beads: `O` at
  Level 1 with `X` resting on top at Level 2.
- **Front-centre peg.** A single `X` sitting on the floor (Level 1) of the Front
  pane's centre column.
- **Back-left peg.** Two `X`s, occupying Level 1 and Level 2.

`render_state` adds a one-line footer with the non-board state (reserves, side
to move, and each player's removal cooldown). For the position above, with
example reserves of 10 each and X to move next:

```text
reserves O=10 X=10  turn=X  cooldown O=False X=False
```

## Machine representation vs. human display

The ASCII above is for humans; the engine stores something else. To maximize
hashing and evaluation speed, `collapse3/game.py` stores the board as a flat
tuple of 9 inner tuples (pegs, read left-to-right, top-to-bottom), each holding
its beads **bottom-first** as integers `0`/`1`, with empty space dropped
entirely. The same position above is:

```python
GameState(
    board=(
        (),       (1,),     (),        # Front pane:  pegs 0, 1, 2
        (0,),     (0, 1),   (0,),      # Middle pane: pegs 3, 4, 5
        (1, 1),   (),       (),        # Back pane:   pegs 6, 7, 8
    ),
    res=(10, 10),                      # remaining reserves (example)
    turn=1,                            # X (player 1) to move
    cooldown=(False, False),           # neither player just removed a bead
)
```

Translation:

- **Peg 4 (Middle-centre)** is `(0, 1)`: `0` on the bottom, `1` on top. The grid
  draws `O` at Level 1, `X` at Level 2, and pads Level 3 with `-`.
- **Peg 6 (Back-left)** is `(1, 1)`: two stacked `X`s.
- **The win.** The engine reads index 0 of pegs 3, 4, 5 and sees all `0`; in the
  notation that is the `O … O … O` line across the Middle pane at Level 1.

To print any state yourself:

```python
from collapse3.render import render_state
print(render_state(state))
```
