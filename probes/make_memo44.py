#!/usr/bin/env python3
"""Generate memo44.pkl: exact labels for every reachable (4,4) state.
~1 minute. Deterministic insertion order (required for split reproduction)."""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collapse3.game import empty_state
from collapse3 import enumeration

memo = enumeration.solve_all(empty_state(4, 4))
out = Path(__file__).resolve().parents[1] / 'memo44.pkl'
with open(out, 'wb') as f:
    pickle.dump(memo, f)
print(f"{out}: {len(memo):,} states")
