import argparse,json,re,sys
from pathlib import Path
import numpy as np
DELIVERY=Path(__file__).resolve().parents[1]
RUNTIME_PATHS=[DELIVERY/'runtime/PaddleFleet/src',DELIVERY/'runtime',DELIVERY/'src']
missing_runtime=[str(p) for p in RUNTIME_PATHS if not p.is_dir()]
if missing_runtime:
 raise RuntimeError(f"Incomplete delivery; missing runtime directories: {missing_runtime}")
for p in RUNTIME_PATHS:
 sys.path.insert(0,str(p))
import paddle
from transformers import AutoTokenizer
from hf_weight_loader import build_target_model,load_hf_safetensors_to_model

def probs(x,temp):
 a=x.cast('float32').numpy().astype(np.float64)/temp;a-=a.max();a=np.exp(a);return a/a.sum()
def draw(p,rng):return int(np.searchsorted(np.cumsum(p),rng.random(),side='right'))
# POSITIONAL uniform for TOKEN-COMMIT draws.
# Sampling itself is unchanged (near-one-hot at low temp, matching SGLang); only the SOURCE of
# the uniform changes: u = f(seed, absolute position) instead of "next value in a stream".
# Why: at low temp `probs()` underflows to exactly one-hot UNLESS two logits are bit-identical,
# in which case mass splits and the draw is a real coin flip.  The target-only loop and the MTP
# loop consume different numbers of stream values (drafts + accept uniforms), so a shared stream
# still desynchronises; keying on position makes both paths break the SAME tie the SAME way.
# Measured: 12 of 18 HF greedy target/mtp mismatches were exact ties (top2_margin=0.000000).
TOKSEED=int(__import__('os').environ.get('TOKSEED','424242'))
# ns namespaces the stream: ns=0 for TARGET-distribution commits (target loop / seed token /
# residual / bonus) so both decode paths use the SAME u at the same position; ns=1 for DRAFT
# draws.  Namespacing matters: draft k and the residual that replaces it land on the same
# absolute position, so a single stream would hand both the identical u and correlate the
# correction sample with the draft it is correcting (invalid speculative sampling for temp>0).
def pos_u(pos,ns=0):return float(np.random.default_rng([TOKSEED,ns,pos]).random())
def draw_at(p,pos,ns=0):
 c=np.cumsum(p);return int(np.searchsorted(c,pos_u(pos,ns)*c[-1],side='right'))
# GREEDY == argmax, matching SpecForge/SGLang rather than sampling a near-one-hot vector.
#   SpecForge  specforge/modeling/draft/dflash.py:23-30 -> `if temperature < 1e-5: argmax`
#   SGLang use specforge/modeling/target/eagle3_target_model.py:554,663
#              -> SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
# Sampling a temp=1e-6 distribution is NOT equivalent: `probs()` underflows to one-hot except
# at bit-identical logits, where mass splits 0.5/0.5 and the draw becomes a coin flip -- which
# is what made the target-only and MTP loops disagree on 12 of 18 HF greedy questions.
# argmax removes the tie by construction (lowest index wins, deterministically, in both paths).
GREEDY_EPS=1e-5
def pick(p,pos,temp,ns=0):
 return int(np.argmax(p)) if (temp is not None and temp<GREEDY_EPS) else draw_at(p,pos,ns)
def last_int(s):
 v=re.findall(r'-?\d[\d,]*',s or '');return int(v[-1].replace(',','')) if v else None
def first_seg(s):return re.split(r'\n\s*\n\s*Question\s*:',s or '',maxsplit=1,flags=re.I)[0]
def forward(model,ids):
 with paddle.no_grad():return model({'input_ids':paddle.to_tensor([ids],dtype='int64')})
def p0(model,context,temp):
 out=forward(model,context+[0]);return probs(out[0].cast('float32')[0,len(context)-1],temp)
_MTP=_EMB=_LMH=None;_bbcap={}
def _setup_chain(model):
 # locate MTP layer / embedding / lm-head; hook lm-head input to capture backbone hidden
 global _MTP,_EMB,_LMH
 for n,l in model.named_sublayers():
  cn=type(l).__name__
  if cn=='MultiTokenPredictionLayer' and _MTP is None:_MTP=l
  if cn=='GPTEmbedding' and _EMB is None:_EMB=l
  if cn.endswith('LMHead') and hasattr(l,'_forward') and _LMH is None:_LMH=l
 def hook(layer,args,kwargs):
  da=args[0] if args else kwargs.get('dict_args')
  if isinstance(da,dict):_bbcap['h']=da.get('hidden_states')
 _LMH.register_forward_pre_hook(hook,with_kwargs=True)
def _embed(id_list):
 t=paddle.to_tensor([id_list],dtype='int64');pos=paddle.arange(len(id_list)).reshape([1,-1]);return _EMB.embedding(t,pos)
def q_chain(model,ids,hid,K,temp,rng):
 # STANDARD spec-decode draft: `hid` is the backbone hidden for positions 0..len(ids)-2, taken from
 # the SAME forward that verified the previous round (FastDeploy: eagle_get_hidden_states over
 # full_hidden_states). No extra target forward per round, and identical sourcing on both frameworks.
 # IN-PLACE depth chaining (verified bit-identical to native out[1]):
 #   depth k: hidden = depth k-1 post-norm output, decoder_input = embeddings shifted by k+1
 hid=hid[:,:len(ids)-1];ds=[];qs=[]
 for k in range(K):
  dec=_embed(ids[k+1:]+ds)                   # shift k+1; len = len(ids)-1
  with paddle.no_grad():
   post=_MTP._proj_and_transformer_layer(hidden_states=hid,decoder_input=dec,rotary_pos_emb=None,attn_mask_startend_row_indices=None)
  lg=_LMH._forward(post[:,-1:,:]).cast('float32')[0,-1]
  q=probs(lg,temp);d=pick(q,len(ids)+k,temp,1);qs.append(q);ds.append(d)
  hid=post
 return ds,qs
def verifier(model,ids,ds,temp):
 # one target forward serves three purposes: verify drafts, give the next round's start token,
 # and give the next round's draft hidden (mirrors FastDeploy's single-forward-per-round design)
 out=forward(model,ids+ds+[0]);lg=out[0].cast('float32')[0];base=len(ids)-1
 ps=[probs(lg[base+i],temp) for i in range(len(ds))];bonus=probs(lg[base+len(ds)],temp)
 return ps,bonus,_bbcap['h'][:1]
def target(model,prompt,max_new,temp,seed):
 rng=np.random.default_rng(seed);seq=list(prompt);gen=[]
 for _ in range(max_new):
  x=pick(p0(model,seq,temp),len(seq),temp);gen.append(x);seq.append(x)
  if x==2:break
 return gen
def mtp(model,prompt,max_new,K,ttemp,dtemp,seed):
 # STANDARD structure: ONE target forward per round. The round's verify forward produces
 #   (a) p for the drafts, (b) the next start token (residual on reject / bonus on full accept),
 #   (c) the hidden the next round's draft consumes.
 # There is NO extra per-round target forward to resample a "seed" token: the start token is
 # whatever the previous verify committed last (FastDeploy: accept_tokens/accept_num +
 # eagle_get_hidden_states). accept_length counts tokens committed per target forward.
 rng=np.random.default_rng(seed);seq=list(prompt);gen=[];acc=[0]*K;den=[0]*K;lead=[];rounds=0
 # prefill: one forward -> first token + hidden for the first draft
 out=forward(model,seq+[0]);lg=out[0].cast('float32')[0];hid=_bbcap['h'][:1]
 first=pick(probs(lg[len(seq)-1],ttemp),len(seq),ttemp);gen.append(first);seq.append(first)
 while len(gen)<max_new:
  room=max_new-len(gen)
  if room<=0:break
  ds,qs=q_chain(model,seq,hid,K,dtemp,rng)
  ps,bp,vh=verifier(model,seq,ds,ttemp)
  commit=[];run=0;rej=False
  for i,(d,q,p) in enumerate(zip(ds,qs,ps)):
   den[i]+=1;alpha=min(1.,p[d]/max(q[d],1e-300))
   if rng.random()<=alpha:commit.append(d);acc[i]+=1;run+=1
   else:
    r=np.maximum(p-q,0);r=r/r.sum() if r.sum()>1e-15 else p
    commit.append(pick(r,len(seq)+len(commit),ttemp));rej=True;break
  if not rej:commit.append(pick(bp,len(seq)+len(commit),ttemp))
  commit=commit[:room];gen.extend(commit);seq.extend(commit);rounds+=1;lead.append(run)
  hid=vh[:,:len(seq)-1]                       # hidden for next draft, from THIS verify forward
  if 2 in commit:gen=gen[:gen.index(2)+1];break
 return gen,{'rounds':rounds,'accepted':acc,'denom':den,'per_step':[acc[i]/den[i] if den[i] else 0 for i in range(K)],'accept_length':1+sum(lead)/len(lead) if lead else 1}
p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--n',type=int,default=4);p.add_argument('--max-new',type=int,default=32);p.add_argument('--K',type=int,default=3);p.add_argument('--temp',type=float,default=1.);p.add_argument('--target-temp',type=float,default=None,help='target model sampling temp (falls back to --temp)');p.add_argument('--draft-temp',type=float,default=None,help='draft/MTP sampling temp (falls back to --temp)');a=p.parse_args();paddle.set_device('gpu:0')
def _fix(t):return 1e-6 if (t is not None and t<=0) else t
TTEMP=_fix(a.target_temp if a.target_temp is not None else a.temp);DTEMP=_fix(a.draft_temp if a.draft_temp is not None else a.temp)
tok=AutoTokenizer.from_pretrained(str(DELIVERY/'tokenizer'),trust_remote_code=True,local_files_only=True);cfg=json.load(open(Path(a.ckpt)/'config.json'));cfg['num_nextn_predict_layers']=1;cfg['use_mtp']=True;cfg['mtp_reuse_last_layer']=True
model=build_target_model(cfg,lazy=False,dtype='bfloat16');stats=load_hf_safetensors_to_model(model,a.ckpt,dtype='bfloat16');model.eval();_setup_chain(model);rows=[json.loads(x) for x in open(a.dataset) if x.strip()][:a.n];samples=[];sa=np.zeros(a.K);sd=np.zeros(a.K);als=[]
import os
if os.path.exists(a.out):
 old=json.load(open(a.out));samples=old.get('samples',[]);sa=np.array(old.get('_sum_acc',[0]*a.K),dtype=float);sd=np.array(old.get('_sum_den',[0]*a.K),dtype=float);als=old.get('_als',[])
start=len(samples)
for i,row in enumerate(rows[start:],start):
 prompt=next(t['value'] for t in row['conversations'] if t['from']=='human');ref=last_int(next(t['value'] for t in row['conversations'] if t['from']=='gpt'));ids=[2]+tok.encode(prompt,add_special_tokens=False);tg=target(model,ids,a.max_new,TTEMP,1000+i);mg,met=mtp(model,ids,a.max_new,a.K,TTEMP,DTEMP,2000+i);sa+=met['accepted'];sd+=met['denom'];als.append(met['accept_length']);to=tok.decode([x for x in tg if x<len(tok)]);mo=tok.decode([x for x in mg if x<len(tok)]);samples.append({'index':i,'reference':ref,'target_ids':tg,'mtp_ids':mg,'target_output':to,'mtp_output':mo,'target_first_prediction':last_int(first_seg(to)),'mtp_first_prediction':last_int(first_seg(mo)),'metrics':met});print(i,'target',samples[-1]['target_first_prediction'],'mtp',samples[-1]['mtp_first_prediction'],'AL',met['accept_length'],flush=True)
 partial={'framework':'native_paddle','algorithm':'standard_rejection_sampling','config':{**vars(a),'resolved_target_temp':TTEMP,'resolved_draft_temp':DTEMP,'sampling':('argmax' if TTEMP<GREEDY_EPS else 'positional_uniform'),'tokseed':TOKSEED},'load_stats':stats,'complete':False,'completed':len(samples),'_sum_acc':sa.tolist(),'_sum_den':sd.tolist(),'_als':als,'per_step_accept':(sa/np.maximum(sd,1)).tolist(),'accept_length':float(np.mean(als)),'samples':samples};tmp=a.out+'.tmp';json.dump(partial,open(tmp,'w'),indent=2,ensure_ascii=False);os.replace(tmp,a.out)
json.dump({'framework':'native_paddle','algorithm':'standard_rejection_sampling','config':{**vars(a),'resolved_target_temp':TTEMP,'resolved_draft_temp':DTEMP,'sampling':('argmax' if TTEMP<GREEDY_EPS else 'positional_uniform'),'tokseed':TOKSEED},'load_stats':stats,'complete':True,'completed':len(samples),'_sum_acc':sa.tolist(),'_sum_den':sd.tolist(),'_als':als,'per_step_accept':(sa/np.maximum(sd,1)).tolist(),'accept_length':float(np.mean(als)),'samples':samples},open(a.out,'w'),indent=2,ensure_ascii=False)
