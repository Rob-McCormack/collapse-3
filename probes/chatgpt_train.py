import sys, time, json, pickle, random, math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collapse3.game import GameState, empty_state, evaluate_terminal, get_legal_moves, apply_move, attrition_value, orient
from collapse3.enumeration import wdl
from collapse3.agents import Agent
from experiments.best_response import solve_best_response, extract_line, grade_line

SEED=314159
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 27 cell-level occupancies: -1 empty, 0=P0, +1=P1 encoded as two signed channels:
# own identity scalar (-1 P0, +1 P1, 0 empty) and occupancy bit; + 2 reserves + turn + 2 cooldown = 33.
def encode_state(s: GameState):
    occ=np.zeros(27,dtype=np.float32)
    who=np.zeros(27,dtype=np.float32)
    for p,peg in enumerate(s.board):
        for z,b in enumerate(peg):
            i=p*3+z; occ[i]=1.0; who[i]=-1.0 if b==0 else 1.0
    return np.concatenate([occ,who,np.array([s.res[0]/4.0,s.res[1]/4.0, float(s.turn), float(s.cooldown[0]), float(s.cooldown[1])],dtype=np.float32)])

DIM=59

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(DIM,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,3),
        )
    def forward(self,x): return self.net(x)


def exact_end(s):
    t=evaluate_terminal(s)
    if t is not None: return t
    if not get_legal_moves(s): return attrition_value(s.board)
    return None

print('loading memo',flush=True)
with open(ROOT/'memo44.pkl','rb') as f: memo=pickle.load(f)
decision=[]; y=[]
for s,v in memo.items():
    if evaluate_terminal(s) is None and get_legal_moves(s):
        decision.append(s)
        y.append(0 if v<0 else (1 if v==0 else 2))
print('decision_states',len(decision),flush=True)

print('encoding training corpus',flush=True)
X=np.empty((len(decision),DIM),dtype=np.float32)
for i,s in enumerate(decision): X[i]=encode_state(s)
y=np.asarray(y,dtype=np.int64)
idx=np.arange(len(decision)); rng=np.random.default_rng(SEED); rng.shuffle(idx)
cut=int(0.2*len(idx)); train_idx=idx[:cut]; test_idx=idx[-int(0.2*len(idx)):]
print('split',len(train_idx),len(test_idx),flush=True)

# Free memo during training; decision states retained for exact holdout audit later.
del memo
model=ValueNet()
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
# Mild class balancing so rare outcome classes are not ignored.
counts=np.bincount(y[train_idx],minlength=3)
weights=counts.sum()/(3*np.maximum(counts,1)); lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
train_ds=TensorDataset(torch.from_numpy(X[train_idx]),torch.from_numpy(y[train_idx]))
loader=DataLoader(train_ds,batch_size=4096,shuffle=True,num_workers=0)
for ep in range(1,9):
    model.train(); total=0; n=0
    for xb,yb in loader:
        opt.zero_grad(); logits=model(xb); loss=lossfn(logits,yb); loss.backward(); opt.step()
        total += float(loss)*len(xb); n+=len(xb)
    model.eval()
    with torch.no_grad():
        # state-value classification sanity check (not policy metric)
        sample=test_idx[:min(50000,len(test_idx))]
        pred=model(torch.from_numpy(X[sample])).argmax(1).numpy()
        acc=float((pred==y[sample]).mean())
    print(f'epoch {ep} loss {total/n:.5f} value_acc_sample {acc:.4f}',flush=True)

torch.save({'state_dict':model.state_dict(),'seed':SEED,'train_fraction':0.2,'dim':DIM},ROOT/'neural_value44.pt')

# Deterministic policy. Neural net estimates P0 WDL expectation of each child.
# Immediate terminals are known from rules and scored exactly, not learned.
class NeuralPolicy(Agent):
    def __init__(self, model):
        self.model=model.eval(); self.name='neural-value-80pct'; self.cache={}
    def choose(self,state):
        if state in self.cache: return self.cache[state]
        moves=get_legal_moves(state)
        scores=[]; infer_children=[]; infer_slots=[]
        for j,m in enumerate(moves):
            c=apply_move(state,m); t=exact_end(c)
            if t is not None:
                scores.append(float(1 if t>0 else (-1 if t<0 else 0)))
            else:
                scores.append(None); infer_children.append(encode_state(c)); infer_slots.append(j)
        if infer_children:
            with torch.no_grad():
                probs=torch.softmax(self.model(torch.from_numpy(np.stack(infer_children))),dim=1).numpy()
            ex=probs[:,2]-probs[:,0]
            for j,v in zip(infer_slots,ex): scores[j]=float(v)
        if state.turn==0:
            k=max(range(len(moves)),key=lambda j:(scores[j],-j))
        else:
            k=min(range(len(moves)),key=lambda j:(scores[j],j))
        self.cache[state]=moves[k]
        return moves[k]

policy=NeuralPolicy(model)

# Exact held-out move audit. Compute on all 20% held-out states using memo reloaded.
print('reloading exact memo for heldout policy audit',flush=True)
with open(ROOT/'memo44.pkl','rb') as f: memo=pickle.load(f)

def exact_move_wdl(s,m): return wdl(memo[apply_move(s,m)],s.turn)

def audit_states(indices):
    total=critical=ok=okcrit=0; reg=regcrit=0; removal_only=rem_n=rem_ok=rem_reg=0
    ordinary_n=ordinary_ok=ordinary_reg=0
    for q,i in enumerate(indices,1):
        s=decision[int(i)]; moves=get_legal_moves(s)
        vals=[exact_move_wdl(s,m) for m in moves]; best=max(vals)
        chosen=policy.choose(s); cv=exact_move_wdl(s,chosen)
        r=best-cv; total+=1; ok+=(r==0); reg+=r
        crit_here=any(v<best for v in vals)
        if crit_here:
            critical+=1; okcrit+=(r==0); regcrit+=r
            optimal=[m for m,v in zip(moves,vals) if v==best]
            ro=all(m[0]=='remove' for m in optimal)
            if ro:
                removal_only+=1; rem_n+=1; rem_ok+=(r==0); rem_reg+=r
            else:
                ordinary_n+=1; ordinary_ok+=(r==0); ordinary_reg+=r
        if q%20000==0: print('audit',q,'/',len(indices),flush=True)
    return {
      'n':total,'optimal_rate':ok/total,'mean_wdl_regret':reg/total,
      'critical_n':critical,'critical_optimal_rate':okcrit/critical,'critical_mean_wdl_regret':regcrit/critical,
      'removal_only_critical_n':rem_n,'removal_only_optimal_rate':rem_ok/rem_n if rem_n else None,'removal_only_mean_regret':rem_reg/rem_n if rem_n else None,
      'other_critical_n':ordinary_n,'other_critical_optimal_rate':ordinary_ok/ordinary_n if ordinary_n else None,'other_critical_mean_regret':ordinary_reg/ordinary_n if ordinary_n else None,
    }

heldout=audit_states(test_idx)
print('heldout',json.dumps(heldout,indent=2),flush=True)

# Exact adversarial certification from empty (4,4), both seats.
rows=[]
for seat in (0,1):
    print('best_response seat',seat,flush=True)
    policy.cache.clear(); st=time.time()
    worst,depth,nstates,brmemo=solve_best_response(policy,4,4,seat)
    row={'seat':seat,'worst_wdl':worst,'worst_case':{1:'win',0:'draw',-1:'forced_loss'}[worst],
         'depth_to_worst':depth,'one_sided_states':nstates,'seconds':time.time()-st,'policy_cache_states':len(policy.cache)}
    line=extract_line(policy,brmemo,4,4,seat)
    row['line']=[{'mover':p,'move':list(m)} for p,m in line]
    row['grading']=grade_line(line,memo,4,4,seat)
    rows.append(row)
    print('RESULT seat',seat,row['worst_case'],'depth',depth,'states',nstates,'sec',row['seconds'],'first_regret',row['grading']['first_policy_positive_regret_ply'],flush=True)

out={
 'experiment':'neural_generalization_exact_best_response',
 'setup':{
  'reserves':[4,4],'seed':SEED,'decision_states':len(decision),'train_fraction':0.2,'train_states':len(train_idx),'heldout_states':len(test_idx),
  'architecture':'59 -> 128 -> 128 -> 64 -> 3-class P0 WDL','epochs':12,
  'note':'Independent reproduction inspired by KIMI 3 description; KIMI neural code was not present in uploaded repo.'
 },
 'heldout':heldout,
 'best_response':rows,
}
(ROOT/'results/neural_best_response_latest.json').write_text(json.dumps(out,indent=2))
print('saved',ROOT/'results/neural_best_response_latest.json',flush=True)
