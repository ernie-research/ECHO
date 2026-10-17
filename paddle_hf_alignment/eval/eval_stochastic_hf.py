import argparse,json,os,sys,re
from pathlib import Path
import numpy as np
import torch
DELIVERY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(DELIVERY/'src'));sys.path.insert(0,str(DELIVERY/'eval'))
from transformers import AutoTokenizer
from modeling.torch_weight_loader import build_and_load_torch
from patch_baseline_mtp import patch_baseline_mtp_weights
def cmask(T, dev, d):
 return torch.triu(torch.full((1,1,T,T),float("-inf"),device=dev,dtype=d),diagonal=1)

def probs(logits,temp):
 x=(logits.float()/temp); x=x-x.max(); return torch.softmax(x,dim=-1).cpu().numpy().astype(np.float64)
def draw(dist,rng):
 c=np.cumsum(dist); return int(np.searchsorted(c,rng.random()*c[-1],side='right'))
# POSITIONAL uniform for TOKEN-COMMIT draws -- see the matching comment in
# eval_stochastic_paddle.py.  Sampling stays near-one-hot (SGLang-compatible); only the source
# of the uniform changes to u=f(seed, absolute position), so the target-only loop and the MTP
# loop break identical logit ties identically.  Same TOKSEED in both evaluators, so the two
# FRAMEWORKS also break ties the same way.
TOKSEED=int(os.environ.get('TOKSEED','424242'))
def pos_u(pos,ns=0):return float(np.random.default_rng([TOKSEED,ns,pos]).random())
def draw_at(dist,pos,ns=0):
 c=np.cumsum(dist); return int(np.searchsorted(c,pos_u(pos,ns)*c[-1],side='right'))
# GREEDY == argmax, matching SpecForge/SGLang (see eval_stochastic_paddle.py for the citations:
# dflash.py:23-30 `if temperature < 1e-5: argmax`, and SamplingParams(temperature=0, top_k=1)).
GREEDY_EPS=1e-5
def pick(dist,pos,temp,ns=0):
 return int(np.argmax(dist)) if (temp is not None and temp<GREEDY_EPS) else draw_at(dist,pos,ns)
def last_int(s):
 v=re.findall(r'-?\d[\d,]*',s or '');return int(v[-1].replace(',','')) if v else None
def first_seg(s): return re.split(r'\n\s*\n\s*Question\s*:',s or '',maxsplit=1,flags=re.I)[0]
def target_logits(model,lm,ids):
 t=torch.tensor([ids],device='cuda'); h,bb,_=model.model(t,use_cache=False,return_mtp_hidden=True);return lm(h).float()[0,-1],bb

def q_chain(model,lm,ids,hid,K,temp,rng):
 # STANDARD spec-decode draft: `hid` is the backbone hidden for positions 0..len(ids)-2, taken from
 # the SAME forward that verified the previous round -- identical sourcing to the paddle side.
 # Previously this reused target_logits(seq)'s hidden (a DIFFERENT forward, length m-1), which under
 # HF's shape-sensitivity (floor 0.5625) put noise between q and p; min(1,p/q) is concave, so that
 # noise systematically LOWERED HF's accept rate at temp=1.0 while leaving greedy (alpha in {0,1})
 # unaffected -- exactly the asymmetry seen in the data.
 # Multi-depth checkpoints (num_nextn_predict_layers>1) carry one head PER DRAFT DEPTH; depth k
 # must use mtp_heads[k].  Even when the heads share every Parameter (Echo mtp3/mtp1.5, where the
 # exported layers 24/25/26 are bit-identical) the router correction bias is per-depth and differs
 # (max|d| up to 0.21), so reusing head 0 at every depth is NOT equivalent.  Single-head
 # checkpoints keep the old behaviour: head 0 is re-applied at every depth.
 heads=model.model.mtp_heads;embed=model.model.embed_tokens
 hid=hid[:,:len(ids)-1];drafts=[];qs=[]
 for k in range(K):
  mtp=heads[k] if k<len(heads) else heads[-1]
  toks=ids[k+1:]+drafts                      # shift k+1; len = len(ids)-1
  t=torch.tensor([toks],device='cuda')
  fused=mtp.eh_proj(torch.cat([mtp.enorm(embed(t)),mtp.hnorm(hid)],-1))
  L=fused.shape[1];cos,sin=model.model.rotary_emb(L,t.device)
  out,_=mtp.transformer_layer(fused,cos,sin,cmask(L,t.device,fused.dtype),None,False)
  q=probs(lm(mtp.shared_head['norm'](out[:,-1:])).float()[0,-1],temp)
  d=pick(q,len(ids)+k,temp,1);drafts.append(d);qs.append(q)
  hid=mtp.shared_head['norm'](out)
 return drafts,qs

def verifier_ps(model,lm,ids,drafts,temp):
 # one target forward per round: verifies drafts, yields the next start token, and yields the
 # hidden the next round's draft consumes (mirrors FastDeploy's single-forward-per-round design)
 t=torch.tensor([ids+drafts],device='cuda');h,bb,_=model.model(t,use_cache=False,return_mtp_hidden=True)
 lg=lm(h).float()[0];base=len(ids)-1
 return [probs(lg[base+i],temp) for i in range(len(drafts))], probs(lg[base+len(drafts)],temp), bb

def run_mtp(model,lm,prompt,max_new,K,ttemp,dtemp,seed):
 # STANDARD structure: ONE target forward per round (see q_chain / verifier_ps comments).
 # No extra per-round forward to resample a "seed" token -- the start token is whatever the
 # previous verify committed last (residual on reject, bonus on full accept).
 rng=np.random.default_rng(seed);seq=list(prompt);gen=[];rounds=0;accepted=[0]*K;den=[0]*K;leading=[];dropped=0
 with torch.no_grad(): t=torch.tensor([seq],device='cuda');h,bb,_=model.model(t,use_cache=False,return_mtp_hidden=True);lg=lm(h).float()[0]
 first=pick(probs(lg[-1],ttemp),len(seq),ttemp);gen.append(first);seq.append(first);hid=bb
 while len(gen)<max_new:
  room=max_new-len(gen)
  if room<=0:break
  with torch.no_grad():
   drafts,qs=q_chain(model,lm,seq,hid,K,dtemp,rng)
   ps,bonus_p,vh=verifier_ps(model,lm,seq,drafts,ttemp)
  acc0=accepted[:];den0=den[:]                  # snapshot: a max_new-truncated round is not counted
  commit=[];run=0;rejected=False
  for i,(d,q,p) in enumerate(zip(drafts,qs,ps)):
   den[i]+=1; alpha=min(1.0,p[d]/max(q[d],1e-300));u=rng.random()
   if u<=alpha:
    commit.append(d);run+=1;accepted[i]+=1
   else:
    residual=np.maximum(p-q,0.0);s_=residual.sum(); residual=(residual/s_) if s_>1e-15 else p
    commit.append(pick(residual,len(seq)+len(commit),ttemp));rejected=True;break
  if not rejected: commit.append(pick(bonus_p,len(seq)+len(commit),ttemp))
  # A round cut short by max_new has fewer commit slots than drafts, so its accept counts are
  # censored (a draft that WOULD have been accepted never gets the chance) and it drags AL down on
  # exactly the tasks with the most truncation. Keep the tokens (accuracy still needs them) but
  # roll the round's metrics back so AL/per-depth only ever see complete rounds.
  if len(commit)>room:
   accepted[:]=acc0;den[:]=den0;dropped+=1;gen.extend(commit[:room]);break
  gen.extend(commit);seq.extend(commit);rounds+=1;leading.append(run)
  hid=vh[:,:len(seq)-1]                       # hidden for next draft, from THIS verify forward
  stop=[t for t in commit if t in STOP_IDS]
  if stop:
   cut=gen.index(stop[0])+1;gen=gen[:cut];break
 return gen,{'rounds':rounds,'accepted':accepted,'denom':den,'dropped_rounds':dropped,'per_step':[accepted[i]/den[i] if den[i] else 0 for i in range(K)],'accept_length':1+sum(leading)/len(leading) if leading else 1}
def run_target(model,lm,prompt,max_new,temp,seed):
 rng=np.random.default_rng(seed);seq=list(prompt);gen=[]
 for _ in range(max_new):
  with torch.no_grad(): lg,_=target_logits(model,lm,seq)
  x=pick(probs(lg,temp),len(seq),temp);gen.append(x);seq.append(x)
  if x in STOP_IDS:break
 return gen
p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--n',type=int,default=4);p.add_argument('--max-new',type=int,default=32);p.add_argument('--K',type=int,default=3);p.add_argument('--temp',type=float,default=1.0);p.add_argument('--target-temp',type=float,default=None);p.add_argument('--draft-temp',type=float,default=None);p.add_argument('--chat-format',choices=['none','target_chatv2'],default='none',help='target_chatv2 wraps the prompt exactly as the SFT training data does and stops on <|im_end|>; required for the *_sft checkpoints')
p.add_argument('--skip-target',action='store_true',help='do not run the extra target-only pass. That pass exists only to check that speculative decoding reproduces plain target decoding, which is already established: greedy target_ids==mtp_ids on 25-35/50 questions, the rest diverging deep into the sequence (median first difference at token 31-138) through bf16 MoE top-8 flips, at a uniform 1.9-3.6 divergences per 1000 agreed tokens. It is accuracy-neutral (target vs mtp correct counts differ by <=2/50), so skipping it only costs the sanity check and halves the cost per question.')
p.add_argument('--extract',choices=['last_int','boxed'],default='last_int',help='answer extraction for the accuracy columns. "boxed" uses the same rules as eval_specbench_prompt.py (\\boxed{} first), so one run yields BOTH acceptance length and accuracy; "last_int" keeps the older behaviour.')
a=p.parse_args()
# Chat template recovered from the SFT h5 data (ids are shown as <target_mask:N> by the local
# tokenizer; the chatv2 tokenizer names them <|im_start|>/<|im_end|>/</think>). Every training
# sample starts with id 2, assistant turns open with the marker (lossmask 0 -> prompt side) and
# close with <|im_end|> (lossmask 1 -> the model emits it, so it is a stop token).
IM_START,IM_END,THINK_CLOSE,ROLE_USER,ROLE_ASSISTANT,NEWLINE=7,8,20,3946,100497,1401
STOP_IDS=frozenset({2,IM_END}) if a.chat_format=='target_chatv2' else frozenset({2})
def build_prompt_ids(text):
 if a.chat_format=='target_chatv2':
  return [2,IM_START,ROLE_USER,NEWLINE]+tok.encode(text,add_special_tokens=False)+[IM_END,NEWLINE,IM_START,ROLE_ASSISTANT,NEWLINE,THINK_CLOSE]
 return [2]+tok.encode(text,add_special_tokens=False)
def _fix(t):return 1e-6 if (t is not None and t<=0) else t
# Grading shares eval/answer_extract.py with eval_specbench_prompt.py, so one spec-decode run can
# report acceptance length AND accuracy under identical rules instead of needing a second pass.
if a.extract=='boxed':
 from answer_extract import extract_str as _extract,first_seg as _fseg,ref_answer_str as _refans,norm_answer as _norm
 def grade(t):return _extract(_fseg(t))
 def grade_ref(g):return _refans(g)
 def is_hit(p,ref):return p is not None and ref is not None and _norm(p)==_norm(ref)
else:
 def grade(t):return last_int(first_seg(t)),'last_int'
 def grade_ref(g):return last_int(g)
 def is_hit(p,ref):return p is not None and ref is not None and p==ref
TTEMP=_fix(a.target_temp if a.target_temp is not None else a.temp);DTEMP=_fix(a.draft_temp if a.draft_temp is not None else a.temp)
torch.cuda.set_device(a.gpu);tok=AutoTokenizer.from_pretrained(str(DELIVERY/'tokenizer'),trust_remote_code=True,local_files_only=True)
model,cfg,stats=build_and_load_torch(a.ckpt,device=f'cuda:{a.gpu}',dtype=torch.bfloat16);patch=patch_baseline_mtp_weights(model,a.ckpt,24,True);model.eval();lm=model.lm_head
rows=[json.loads(x) for x in open(a.dataset) if x.strip()][:a.n];samples=[];sum_acc=np.zeros(a.K);sum_den=np.zeros(a.K);als=[]
if os.path.exists(a.out):
 old=json.load(open(a.out));samples=old.get('samples',[]);sum_acc=np.array(old.get('_sum_acc',[0]*a.K),dtype=float);sum_den=np.array(old.get('_sum_den',[0]*a.K),dtype=float);als=old.get('_als',[])
start=len(samples)
for i,row in enumerate(rows[start:],start):
 prompt=next(t['value'] for t in row['conversations'] if t['from']=='human');gold=next(t['value'] for t in row['conversations'] if t['from']=='gpt');ref=grade_ref(gold);ids=build_prompt_ids(prompt)
 tgt=[] if a.skip_target else run_target(model,lm,ids,a.max_new,TTEMP,1000+i);mtp,met=run_mtp(model,lm,ids,a.max_new,a.K,TTEMP,DTEMP,2000+i);sum_acc+=met['accepted'];sum_den+=met['denom'];als.append(met['accept_length'])
 to=tok.decode([x for x in tgt if x<len(tok)]);mo=tok.decode([x for x in mtp if x<len(tok)]);tp,tr=grade(to);mp,mr=grade(mo);samples.append({'index':i,'reference':ref,'target_ids':tgt,'mtp_ids':mtp,'target_output':to,'mtp_output':mo,'target_first_prediction':tp,'mtp_first_prediction':mp,'target_rule':tr,'mtp_rule':mr,'mtp_correct':is_hit(mp,ref),'target_correct':is_hit(tp,ref),'metrics':met});print(i,'target',tp,'mtp',mp,'ref',ref,'AL',met['accept_length'],'mtp_acc',sum(1 for s in samples if s['mtp_correct']),'/',len(samples),flush=True)
 acc_fields={'mtp_correct':sum(1 for s in samples if s['mtp_correct']),'target_correct':sum(1 for s in samples if s['target_correct']),'graded':len(samples),'extract':a.extract}
 partial={'framework':'hf_torch','algorithm':'standard_rejection_sampling','config':{**vars(a),'resolved_target_temp':TTEMP,'resolved_draft_temp':DTEMP,'sampling':('argmax' if TTEMP<GREEDY_EPS else 'positional_uniform'),'tokseed':TOKSEED,'attn_impl':os.environ.get('TARGET_ATTN','eager')},'load_stats':stats,'patch_stats':patch,'complete':False,'completed':len(samples),'accuracy':acc_fields,'_sum_acc':sum_acc.tolist(),'_sum_den':sum_den.tolist(),'_als':als,'per_step_accept':(sum_acc/np.maximum(sum_den,1)).tolist(),'accept_length':float(np.mean(als)),'samples':samples};tmp=a.out+'.tmp';json.dump(partial,open(tmp,'w'),indent=2,ensure_ascii=False);os.replace(tmp,a.out)
out={'framework':'hf_torch','algorithm':'standard_rejection_sampling','config':{**vars(a),'resolved_target_temp':TTEMP,'resolved_draft_temp':DTEMP,'sampling':('argmax' if TTEMP<GREEDY_EPS else 'positional_uniform'),'tokseed':TOKSEED,'attn_impl':os.environ.get('TARGET_ATTN','eager')},'load_stats':stats,'patch_stats':patch,'complete':True,'completed':len(samples),'accuracy':{'mtp_correct':sum(1 for s in samples if s['mtp_correct']),'target_correct':sum(1 for s in samples if s['target_correct']),'graded':len(samples),'extract':a.extract},'_sum_acc':sum_acc.tolist(),'_sum_den':sum_den.tolist(),'_als':als,'per_step_accept':(sum_acc/np.maximum(sum_den,1)).tolist(),'accept_length':float(np.mean(als)),'samples':samples};json.dump(out,open(a.out,'w'),indent=2,ensure_ascii=False)
print('DONE AL=%.4f mtp_acc=%d/%d target_acc=%d/%d per_step=%s'%(float(np.mean(als)),out['accuracy']['mtp_correct'],len(samples),out['accuracy']['target_correct'],len(samples),[round(x,3) for x in out['per_step_accept']]))
