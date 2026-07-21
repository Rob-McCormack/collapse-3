#!/usr/bin/env python3
"""Generalization probe (Kimi setup): raw-value regression, graded exactly.

  1. Enumerate (3,3) and (4,4) exactly (solve_all).
  2. Train MLP value nets on fractions of (4,4) decision states; grade 1-ply
     play on held-out states EXACTLY (optimal-move rate, WDL regret; all and
     critical-only). Baselines: random (exact), hand-crafted positional 1-ply.
  3. Cross-size transfer: train on all (3,3), grade on held-out (4,4).
  4. Extrapolation: train (3,3)+(4,4), grade on (5,5) states sampled by
     random playouts and labeled exactly with the alpha-beta solver.
  5. Failure anatomy: errors split by removal-only-optimal states.

Features: 27 cells x {empty,P0,P1} one-hot + turn + reserves(/14) + cooldown.
Decision states: non-terminal, >=2 legal moves (repo census uses >=1: 477,960
at (4,4); the >=2 subset is 450,675).
Leakage note: random splits place children of test states in train at high
fractions; at 0.5% train this is negligible. Value targets only.
Requires: torch.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch, torch.nn as nn
from collapse3 import enumeration, agents
from collapse3.game import (empty_state, get_legal_moves, apply_move,
                            evaluate_terminal, orient)
from collapse3 import solver as sv

t_start = time.time()
def log(msg): print(f"[{time.time()-t_start:6.0f}s] {msg}", flush=True)

def featurize(s):
    f = np.zeros(86, dtype=np.float32)
    for peg in range(9):
        stack = s.board[peg]
        for z in range(3):
            base = peg * 9 + z * 3
            if z < len(stack): f[base + 1 + stack[z]] = 1.0
            else: f[base] = 1.0
    f[81] = float(s.turn); f[82] = s.res[0] / 14.0; f[83] = s.res[1] / 14.0
    f[84] = float(s.cooldown[0]); f[85] = float(s.cooldown[1])
    return f

def build_dataset(memo):
    X, y, S, C = [], [], [], []
    for s, v in memo.items():
        if evaluate_terminal(s) is not None: continue
        if len(get_legal_moves(s)) < 2: continue
        X.append(featurize(s)); y.append(v); S.append(s)
        vals = set(enumeration.action_values(memo, s, unit='wdl').values())
        C.append(len(vals) > 1)
    return np.array(X), np.array(y, dtype=np.float32), S, np.array(C, dtype=bool)

log("enumerating (3,3)"); memo33 = enumeration.solve_all(empty_state(3, 3))
log(f"  {len(memo33):,} states")
log("enumerating (4,4)"); memo44 = enumeration.solve_all(empty_state(4, 4))
log(f"  {len(memo44):,} states")
log("building datasets")
X33, y33, S33, C33 = build_dataset(memo33)
X44, y44, S44, C44 = build_dataset(memo44)
log(f"  (3,3): {len(X33):,} decision states | (4,4): {len(X44):,} ({C44.sum():,} critical)")

rng = np.random.default_rng(0)
perm = rng.permutation(len(X44))
n_test = int(0.2 * len(X44))
test_idx, train_idx = perm[:n_test], perm[n_test:]
S_test = [S44[i] for i in test_idx]
crit_test = C44[test_idx]

def train_net(X, y, epochs=6, bs=4096, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(86, 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(X); yt = torch.tensor(y / 100.0).unsqueeze(1)
    n = len(Xt)
    for _ in range(epochs):
        p = torch.randperm(n)
        for i in range(0, n, bs):
            idx = p[i:i+bs]
            loss = nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def grade(net, memo, states):
    res = np.zeros((len(states), 3), dtype=np.float32)
    feats, refs = [], []
    for i, s in enumerate(states):
        for m in get_legal_moves(s):
            c = apply_move(s, m)
            t = evaluate_terminal(c)
            if t is None:
                feats.append(featurize(c)); refs.append((i, m, None))
            else:
                refs.append((i, m, float(t)))
    preds = iter(net(torch.tensor(np.array(feats))).numpy().flatten() * 100.0) if feats else iter([])
    from collections import defaultdict
    per = defaultdict(list)
    for (i, m, t) in refs:
        per[i].append((m, float(t) if t is not None else float(next(preds))))
    for i, s in enumerate(states):
        mv = per[i]
        chosen = (max if s.turn == 0 else min)(mv, key=lambda x: x[1])[0]
        avw = enumeration.action_values(memo, s, unit='wdl')
        avr = enumeration.action_values(memo, s, unit='raw')
        best_raw = max(avr.values())
        opt_moves = [m for m, v in avr.items() if v == best_raw]
        res[i] = (1.0 if avr[chosen] == best_raw else 0.0,
                  max(avw.values()) - avw[chosen],
                  1.0 if all(m[0] == 'remove' for m in opt_moves) else 0.0)
    return res

def random_baseline(memo, states):
    out = np.zeros((len(states), 2), dtype=np.float32)
    for i, s in enumerate(states):
        vals = np.array(list(enumeration.action_values(memo, s, unit='wdl').values()))
        out[i] = ((vals == vals.max()).mean(), (vals.max() - vals).mean())
    return out

def heuristic_baseline(memo, states, seed=1):
    rng_h = np.random.default_rng(seed)
    out = np.zeros((len(states), 2), dtype=np.float32)
    for i, s in enumerate(states):
        me = s.turn
        scored = []
        for m in get_legal_moves(s):
            c = apply_move(s, m)
            t = evaluate_terminal(c)
            v = float(orient(t, me)) if t is not None else float(agents.board_heuristic(c.board, me))
            scored.append((m, v, rng_h.random()))
        chosen = max(scored, key=lambda x: (x[1], x[2]))[0]
        avw = enumeration.action_values(memo, s, unit='wdl')
        avr = enumeration.action_values(memo, s, unit='raw')
        out[i] = (1.0 if avr[chosen] == max(avr.values()) else 0.0,
                  max(avw.values()) - avw[chosen])
    return out

def report(name, res, crit):
    oa, ra = res[:, 0].mean(), res[:, 1].mean()
    oc, rc = res[crit, 0].mean(), res[crit, 1].mean()
    print(f"{name:>30}: optimal {oa:.3f} (crit {oc:.3f}) | regret {ra:.4f} (crit {rc:.4f})", flush=True)

print("\n=== 1. Train-fraction sweep (within (4,4), held-out 20%) ===", flush=True)
for frac in (0.005, 0.01, 0.05, 0.25, 0.80):
    n_sub = max(500, int(frac * len(train_idx)))
    net = train_net(X44[train_idx[:n_sub]], y44[train_idx[:n_sub]])
    report(f"  net train {n_sub:,} ({frac:.1%})", grade(net, memo44, S_test), crit_test)

log("baselines")
report("  random (exact)", random_baseline(memo44, S_test), crit_test)
report("  hand-crafted positional 1-ply", heuristic_baseline(memo44, S_test), crit_test)

print("\n=== 2. Cross-size transfer: train ALL of (3,3), grade held-out (4,4) ===", flush=True)
net33 = train_net(X33, y33)
report("  net (3,3)->(4,4)", grade(net33, memo44, S_test), crit_test)

print("\n=== 3. Extrapolation: train (3,3)+(4,4), grade on (5,5) sample ===", flush=True)
t0 = time.time()
net34 = train_net(np.concatenate([X33, X44]), np.concatenate([y33, y44]))
log(f"  trained on {len(X33)+len(X44):,} states [{time.time()-t0:.0f}s]")

def sample_states(n_target, seed=7):
    import random as pyrandom
    rng_s = pyrandom.Random(seed)
    seen, out = set(), []
    while len(out) < n_target:
        s = empty_state(5, 5)
        while True:
            if evaluate_terminal(s) is not None: break
            moves = get_legal_moves(s)
            if not moves: break
            if len(moves) >= 2 and s not in seen:
                seen.add(s); out.append(s)
                if len(out) >= n_target: break
            s = apply_move(s, rng_s.choice(moves))
    return out

log("  sampling (5,5) states")
S55 = sample_states(1200)
log("  labeling with exact solver (one shared TT for all solves)")
label_memo = {}
sv.reset_search_state()
def exact_value(s):
    if s not in label_memo:
        label_memo[s] = sv.game_value(s)
    return label_memo[s]

for s in S55:
    exact_value(s)
    for m in get_legal_moves(s):
        exact_value(apply_move(s, m))
log(f"  labeled {len(label_memo):,} states")

res55 = np.zeros((len(S55), 3), dtype=np.float32)
crit55 = np.zeros(len(S55), dtype=bool)
for i, s in enumerate(S55):
    me = s.turn
    scored = []
    for m in get_legal_moves(s):
        c = apply_move(s, m)
        t = evaluate_terminal(c)
        v = float(t) if t is not None else float(net34(torch.tensor(featurize(c)).unsqueeze(0)).item() * 100.0)
        scored.append((m, v))
    chosen = (max if me == 0 else min)(scored, key=lambda x: x[1])[0]
    child_vals = {}
    for m in get_legal_moves(s):
        c = apply_move(s, m)
        t = evaluate_terminal(c)
        child_vals[m] = t if t is not None else label_memo[c]
    raw = {m: orient(v, me) for m, v in child_vals.items()}
    wdlv = {m: (1 if v > 0 else (-1 if v < 0 else 0)) for m, v in raw.items()}
    res55[i, 0] = 1.0 if raw[chosen] == max(raw.values()) else 0.0
    res55[i, 1] = max(wdlv.values()) - wdlv[chosen]
    crit55[i] = len(set(wdlv.values())) > 1
report("  net (3,3)+(4,4)->(5,5)", res55, crit55)
print(f"  (5,5) sample: {len(S55)} states, {crit55.sum()} critical", flush=True)

print("\n=== 4. Where the net fails (within (4,4), 80% net) ===", flush=True)
n80 = int(0.8 * len(train_idx))
net80 = train_net(X44[train_idx[:n80]], y44[train_idx[:n80]])
res80 = grade(net80, memo44, S_test)
rem_opt = res80[:, 2].astype(bool)
crit_rem = crit_test & rem_opt
crit_nrem = crit_test & ~rem_opt
print(f"  critical, removal-only-optimal ({crit_rem.sum():,}): "
      f"optimal {res80[crit_rem,0].mean():.3f}, regret {res80[crit_rem,1].mean():.4f}", flush=True)
print(f"  critical, other states         ({crit_nrem.sum():,}): "
      f"optimal {res80[crit_nrem,0].mean():.3f}, regret {res80[crit_nrem,1].mean():.4f}", flush=True)
log("done")
