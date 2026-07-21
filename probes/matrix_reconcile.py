#!/usr/bin/env python3
"""Reconciliation matrix: whose errors live on removal-only states?

Setups:
  A (ChatGPT): 59-dim encoding, 3-class WDL classifier, class-balanced CE,
               AdamW(1e-3, wd 1e-5), 8 epochs, bs 4096. Policy: expected WDL.
  B (Kimi):    86-dim one-hot, raw-value regressor (v/100), MSE, Adam(1e-3),
               6 epochs, bs 4096, 256-256. Policy: predicted raw value.

Split convention (ChatGPT's): shuffle decision-state indices with the seed;
first 20% train, last 20% test, middle 60% unused.
Grading (identical for all nets): WDL-optimality and WDL regret vs memo.
Analysis buckets (critical states only):
  remWDL: all WDL-optimal moves are removals   (ChatGPT's definition)
  remRAW: all RAW-optimal moves are removals   (Kimi's definition)
Then: best_response certification of each frozen net, both seats, with
provenance (train/test/unused) of the first value-losing state.
Requires: torch, memo44.pkl (run make_memo44.py first).
"""
import sys, time, json, pickle, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1); torch.set_num_interop_threads(1)

from collapse3.game import empty_state, evaluate_terminal, get_legal_moves, apply_move, attrition_value, orient
from collapse3.enumeration import wdl
from collapse3.agents import Agent
from experiments.best_response import solve_best_response, extract_line

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log('loading memo')
with open(ROOT / 'memo44.pkl', 'rb') as f:
    memo = pickle.load(f)
decision = [s for s in memo if evaluate_terminal(s) is None and get_legal_moves(s)]
log(f'{len(decision):,} decision states')

def encode_A(s):
    occ = np.zeros(27, dtype=np.float32); who = np.zeros(27, dtype=np.float32)
    for p, peg in enumerate(s.board):
        for z, b in enumerate(peg):
            i = p * 3 + z; occ[i] = 1.0; who[i] = -1.0 if b == 0 else 1.0
    return np.concatenate([occ, who, np.array([s.res[0]/4., s.res[1]/4., float(s.turn),
                          float(s.cooldown[0]), float(s.cooldown[1])], np.float32)])

def encode_B(s):
    f = np.zeros(86, dtype=np.float32)
    for peg in range(9):
        stack = s.board[peg]
        for z in range(3):
            base = peg * 9 + z * 3
            if z < len(stack): f[base + 1 + stack[z]] = 1.0
            else: f[base] = 1.0
    f[81] = float(s.turn); f[82] = s.res[0]/14.; f[83] = s.res[1]/14.
    f[84] = float(s.cooldown[0]); f[85] = float(s.cooldown[1])
    return f

class NetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(59,128), nn.ReLU(), nn.Linear(128,128),
                                 nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,3))
    def forward(self, x): return self.net(x)

def train_A(Xtr, ytr_cls, seed):
    torch.manual_seed(seed)
    model = NetA()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    counts = np.bincount(ytr_cls, minlength=3)
    weights = counts.sum() / (3 * np.maximum(counts, 1))
    lossfn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr_cls)
    n = len(Xt)
    for ep in range(8):
        p = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = p[i:i+4096]
            loss = lossfn(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

def train_B(Xtr, ytr_raw, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(86,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(Xtr); yt = torch.tensor(ytr_raw / 100.0).unsqueeze(1)
    n = len(Xt)
    for ep in range(6):
        p = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = p[i:i+4096]
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

def exact_end(s):
    t = evaluate_terminal(s)
    if t is not None: return t
    if not get_legal_moves(s): return attrition_value(s.board)
    return None

class PolicyA(Agent):
    def __init__(self, model):
        self.model = model; self.name = 'A-wdl-clf'; self.cache = {}
    def choose(self, state):
        got = self.cache.get(state)
        if got is not None: return got
        moves = get_legal_moves(state); scores = [None]*len(moves); feats = []; slots = []
        for j, m in enumerate(moves):
            c = apply_move(state, m); t = exact_end(c)
            if t is not None: scores[j] = 1. if t > 0 else (-1. if t < 0 else 0.)
            else: feats.append(encode_A(c)); slots.append(j)
        if feats:
            with torch.inference_mode():
                probs = torch.softmax(self.model(torch.from_numpy(np.stack(feats))), 1).numpy()
            ex = probs[:,2] - probs[:,0]
            for j, v in zip(slots, ex): scores[j] = float(v)
        k = (max if state.turn == 0 else min)(range(len(moves)), key=lambda j: (scores[j], -j if state.turn==0 else j))
        self.cache[state] = moves[k]; return moves[k]

class PolicyB(Agent):
    def __init__(self, model):
        self.model = model; self.name = 'B-raw-reg'; self.cache = {}
    def choose(self, state):
        got = self.cache.get(state)
        if got is not None: return got
        moves = get_legal_moves(state); scores = [None]*len(moves); feats = []; slots = []
        for j, m in enumerate(moves):
            c = apply_move(state, m); t = exact_end(c)
            if t is not None: scores[j] = float(t)
            else: feats.append(encode_B(c)); slots.append(j)
        if feats:
            with torch.inference_mode():
                preds = self.model(torch.from_numpy(np.stack(feats))).numpy().flatten() * 100.0
            for j, v in zip(slots, preds): scores[j] = float(v)
        k = (max if state.turn == 0 else min)(range(len(moves)), key=lambda j: (scores[j], -j if state.turn==0 else j))
        self.cache[state] = moves[k]; return moves[k]

def move_labels(s):
    moves = get_legal_moves(s)
    w = [wdl(memo[apply_move(s, m)], s.turn) for m in moves]
    r = [orient(memo[apply_move(s, m)], s.turn) for m in moves]
    return moves, w, r

def audit(policy, states):
    out = dict(n=0, ok=0, reg=0, crit=0, okc=0, regc=0,
               rw=dict(n=0, ok=0, reg=0), rr=dict(n=0, ok=0, reg=0), other=dict(n=0, ok=0, reg=0))
    for s in states:
        moves, w, r = move_labels(s)
        bestw = max(w); bestr = max(r)
        chosen = policy.choose(s)
        ci = moves.index(chosen)
        reg = bestw - w[ci]
        out['n'] += 1; out['ok'] += (reg == 0); out['reg'] += reg
        if any(v < bestw for v in w):
            out['crit'] += 1; out['okc'] += (reg == 0); out['regc'] += reg
            remWDL = all(m[0] == 'remove' for m, v in zip(moves, w) if v == bestw)
            remRAW = all(m[0] == 'remove' for m, v in zip(moves, r) if v == bestr)
            for key, flag in (('rw', remWDL), ('rr', remRAW)):
                if flag:
                    out[key]['n'] += 1; out[key]['ok'] += (reg == 0); out[key]['reg'] += reg
            if not remWDL and not remRAW:
                out['other']['n'] += 1; out['other']['ok'] += (reg == 0); out['other']['reg'] += reg
    def rate(d): return (d['ok']/d['n'] if d['n'] else None, d['reg']/d['n'] if d['n'] else None)
    o, rg = rate(out); oc, rgc = rate(dict(ok=out['okc'], reg=out['regc'], n=out['crit']))
    return {'overall': (o, rg), 'critical': (oc, rgc),
            'remWDL': rate(out['rw']), 'remRAW': rate(out['rr']), 'other': rate(out['other']),
            'counts': {'n': out['n'], 'crit': out['crit'], 'remWDL': out['rw']['n'],
                       'remRAW': out['rr']['n'], 'other': out['other']['n']}}

def first_error_provenance(policy, seat, split_of):
    pol = PolicyA(policy.model) if isinstance(policy, PolicyA) else PolicyB(policy.model)
    worst, depth, nstates, brmemo = solve_best_response(pol, 4, 4, seat)
    line = extract_line(pol, brmemo, 4, 4, seat)
    s = empty_state(4, 4)
    prov = None; ply_of_error = None
    for ply, (mover, m) in enumerate(line, 1):
        if mover == seat and prov is None:
            moves, w, r = move_labels(s)
            if w[moves.index(m)] < max(w):
                prov = split_of.get(s, 'unused-60%'); ply_of_error = ply
        s = apply_move(s, m)
    return worst, depth, prov, ply_of_error

results = []
for seed in (314159, 0, 1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    idx = np.arange(len(decision)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    n20 = int(0.2 * len(idx))
    train_idx, test_idx = idx[:n20], idx[-n20:]
    split_of = {}
    for i in train_idx: split_of[decision[int(i)]] = 'train'
    for i in test_idx: split_of[decision[int(i)]] = 'test'
    S_test = [decision[int(i)] for i in test_idx]

    for setup in ('A', 'B'):
        log(f'seed {seed} setup {setup}: encoding train')
        if setup == 'A':
            Xtr = np.stack([encode_A(decision[int(i)]) for i in train_idx])
            ytr = np.array([0 if memo[decision[int(i)]] < 0 else (1 if memo[decision[int(i)]] == 0 else 2)
                            for i in train_idx], dtype=np.int64)
            model = train_A(Xtr, ytr, seed); pol = PolicyA(model)
        else:
            Xtr = np.stack([encode_B(decision[int(i)]) for i in train_idx])
            ytr = np.array([memo[decision[int(i)]] for i in train_idx], dtype=np.float32)
            model = train_B(Xtr, ytr, seed); pol = PolicyB(model)
        log(f'seed {seed} setup {setup}: audit on {len(S_test):,} held-out')
        rep = audit(pol, S_test)
        certs = {}
        for seat in (0, 1):
            worst, depth, prov, ply_err = first_error_provenance(pol, seat, split_of)
            certs[seat] = {'worst': worst, 'depth': depth, 'first_error_state': prov, 'ply': ply_err}
            log(f'  seat {seat}: worst={worst} depth={depth} first-error={prov} at ply {ply_err}')
        results.append({'seed': seed, 'setup': setup, 'audit': rep, 'cert': certs})
        log(f"  overall {rep['overall'][0]:.4f} crit {rep['critical'][0]:.4f} | "
            f"remWDL {rep['remWDL'][0]} (n={rep['counts']['remWDL']}) "
            f"remRAW {rep['remRAW'][0]} (n={rep['counts']['remRAW']}) "
            f"other {rep['other'][0]} (n={rep['counts']['other']})")

with open(ROOT / 'results' / 'matrix.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
log('saved results/matrix.json')

print("\n================ MATRIX SUMMARY ================")
for r in results:
    a = r['audit']
    print(f"seed {r['seed']:>6} setup {r['setup']}: overall {a['overall'][0]:.4f} crit {a['critical'][0]:.4f} "
          f"| remWDL {a['remWDL'][0]:.4f} (n={a['counts']['remWDL']}) "
          f"| remRAW {a['remRAW'][0]:.4f} (n={a['counts']['remRAW']}) "
          f"| other {a['other'][0]:.4f} (n={a['counts']['other']}) "
          f"| cert s0 {r['cert'][0]['worst']}/d{r['cert'][0]['depth']} s1 {r['cert'][1]['worst']}/d{r['cert'][1]['depth']}")
