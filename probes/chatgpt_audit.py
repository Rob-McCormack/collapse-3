import sys,time,json,pickle,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from collapse3.game import GameState, empty_state, evaluate_terminal, get_legal_moves, apply_move, attrition_value
from collapse3.enumeration import wdl
from collapse3.agents import Agent
from experiments.best_response import solve_best_response, extract_line, grade_line
SEED=314159

def encode_state(s):
    occ=np.zeros(27,dtype=np.float32); who=np.zeros(27,dtype=np.float32)
    for p,peg in enumerate(s.board):
        for z,b in enumerate(peg):
            i=p*3+z; occ[i]=1.; who[i]=-1. if b==0 else 1.
    return np.concatenate([occ,who,np.array([s.res[0]/4.,s.res[1]/4.,float(s.turn),float(s.cooldown[0]),float(s.cooldown[1])],np.float32)])
DIM=59
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x): return self.net(x)

def exact_end(s):
    t=evaluate_terminal(s)
    if t is not None:return t
    if not get_legal_moves(s):return attrition_value(s.board)
    return None

ck=torch.load(ROOT/'neural_value44.pt',map_location='cpu',weights_only=False); model=ValueNet(); model.load_state_dict(ck['state_dict']); model.eval()

class NeuralPolicy(Agent):
    def __init__(self,model):self.model=model;self.name='neural-value-20pct';self.cache={}
    def choose(self,state):
        got=self.cache.get(state)
        if got is not None:return got
        moves=get_legal_moves(state); scores=[None]*len(moves); feats=[]; slots=[]
        for j,m in enumerate(moves):
            c=apply_move(state,m); t=exact_end(c)
            if t is not None:scores[j]=1. if t>0 else (-1. if t<0 else 0.)
            else:feats.append(encode_state(c));slots.append(j)
        if feats:
            with torch.inference_mode():
                probs=torch.softmax(self.model(torch.from_numpy(np.stack(feats))),1).numpy()
            ex=probs[:,2]-probs[:,0]
            for j,v in zip(slots,ex):scores[j]=float(v)
        if state.turn==0:k=max(range(len(moves)),key=lambda j:(scores[j],-j))
        else:k=min(range(len(moves)),key=lambda j:(scores[j],j))
        self.cache[state]=moves[k];return moves[k]
policy=NeuralPolicy(model)

print('loading exact memo',flush=True)
with open(ROOT/'memo44.pkl','rb') as f:memo=pickle.load(f)
decision=[s for s in memo if evaluate_terminal(s) is None and get_legal_moves(s)]
idx=np.arange(len(decision)); rng=np.random.default_rng(SEED);rng.shuffle(idx); n20=int(.2*len(idx)); test_idx=idx[-n20:]
print('decision',len(decision),'holdout',len(test_idx),flush=True)

def audit(indices):
    total=critical=ok=okc=reg=regc=0; rn=rok=rreg=0; on=ook=oreg=0
    for q,i in enumerate(indices,1):
        s=decision[int(i)];moves=get_legal_moves(s);vals=[wdl(memo[apply_move(s,m)],s.turn) for m in moves];best=max(vals)
        ch=policy.choose(s);cv=wdl(memo[apply_move(s,ch)],s.turn);r=best-cv
        total+=1;ok+=r==0;reg+=r
        if any(v<best for v in vals):
            critical+=1;okc+=r==0;regc+=r
            opts=[m for m,v in zip(moves,vals) if v==best]
            if all(m[0]=='remove' for m in opts):rn+=1;rok+=r==0;rreg+=r
            else:on+=1;ook+=r==0;oreg+=r
        if q%20000==0:print('audit',q,flush=True)
    return {'n':total,'optimal_rate':ok/total,'mean_wdl_regret':reg/total,'critical_n':critical,'critical_optimal_rate':okc/critical,'critical_mean_wdl_regret':regc/critical,
            'removal_only_critical_n':rn,'removal_only_optimal_rate':rok/rn if rn else None,'removal_only_mean_regret':rreg/rn if rn else None,
            'other_critical_n':on,'other_critical_optimal_rate':ook/on if on else None,'other_critical_mean_regret':oreg/on if on else None}

st=time.time();held=audit(test_idx);print('HELD',json.dumps(held,indent=2),'sec',time.time()-st,flush=True)
rows=[]
for seat in (0,1):
    policy.cache.clear();print('best_response seat',seat,flush=True);st=time.time();worst,depth,nstates,brmemo=solve_best_response(policy,4,4,seat)
    line=extract_line(policy,brmemo,4,4,seat);grading=grade_line(line,memo,4,4,seat)
    row={'seat':seat,'worst_wdl':worst,'worst_case':{1:'win',0:'draw',-1:'forced_loss'}[worst],'depth_to_worst':depth,'one_sided_states':nstates,'seconds':time.time()-st,'policy_states':len(policy.cache),'line':[{'mover':p,'move':list(m)} for p,m in line],'grading':grading}
    rows.append(row);print('RESULT',json.dumps({k:v for k,v in row.items() if k not in ('line','grading')},indent=2),'first_regret',grading['first_policy_positive_regret_ply'],flush=True)
out={'experiment':'neural_generalization_exact_best_response','setup':{'reserves':[4,4],'seed':SEED,'decision_states':len(decision),'train_fraction':0.2,'train_states':n20,'heldout_states':n20,'architecture':'59->128->128->64->3-class P0 WDL','epochs':8,'note':'Independent reproduction inspired by KIMI 3 description; KIMI neural code/model was not present in the uploaded repo.'},'heldout':held,'best_response':rows}
path=ROOT/'results/neural_best_response_latest.json';path.write_text(json.dumps(out,indent=2));print('saved',path,flush=True)
