"""PIQA and WinoGrande under the protocol the literature actually uses: compare loglikelihoods.

Why this exists
---------------
The platform serves both sets as a 5-shot A/B letter question with `max_tokens=2`. That is
OpenCompass's `gen` path, which its own docs reserve for chat models -- base models are supposed to
use `ppl`. Consequences measured on 2026-09-25:

  * PIQA splits hard by gold position. echo_mtp3 scores 94.1% when the answer sits at A and 32.4%
    when it sits at B (below chance), and across columns the score tracks the A-rate: the two
    columns that answer A ~51% of the time are the two highest scorers.
  * WinoGrande sits at 49.0-54.5 for all seven columns against a chance level of 50, i.e. the row
    carries no signal, while 7B-class models score 70+ under the loglikelihood protocol.

So the row is measuring letter-position preference, not commonsense. This script implements the
lm-evaluation-harness protocol instead, on the SAME items as the platform extraction:

  PIQA (lm_eval/tasks/piqa/piqa.yaml): 0-shot, context is `Question: {goal}\\nAnswer:` and the two
  candidates are the SOLUTION TEXTS. No letters exist, so position preference cannot enter.
  Reports `acc` (summed logprob) and `acc_norm` (per-byte, because PIQA's two solutions differ in
  length and unnormalized logprob favours the shorter one).

  WinoGrande (lm_eval/tasks/winogrande): the candidate is substituted INTO the blank and the shared
  remainder of the sentence is scored. Both branches score the identical continuation, so length
  normalization is a no-op and only `acc` is meaningful. This is the harness's split, not the
  "obvious" one -- the harness runs a few points higher than the obvious split but preserves
  ranking (EleutherAI issue #978).

Usage:
  python eval/eval_mc_loglik.py --bench PIQA_wocot_0shot --ckpt ckpts/ds_mtp1/step_78000/extracted/hf_checkpoint-78000 \
      --cfg ds_mtp1 --gpu 0
Output: results/mc_loglik/<bench>__<cfg>.json
"""
import argparse, glob, json, time
from pathlib import Path
import sys

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D / 'src'))
sys.path.insert(0, str(D / 'eval'))

import tvshim  # noqa: E402,F401  broken torchvision in this venv; see eval/tvshim.py
ap = argparse.ArgumentParser()
ap.add_argument('--bench', required=True, choices=['PIQA_wocot_0shot', 'winogrande_wocot_5shot'])
ap.add_argument('--ckpt', required=True)
ap.add_argument('--cfg', required=True, help='column name, only used for the output filename')
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--limit', type=int, default=0, help='first N items only, for a smoke run')
ap.add_argument('--out', default='')
a = ap.parse_args()
OUT = Path(a.out) if a.out else D / 'results' / 'mc_loglik' / f'{a.bench}__{a.cfg}.json'


def pairs(bench, r):
    """(context, continuation) for each candidate, in the extraction's option order."""
    if bench == 'PIQA_wocot_0shot':
        ctx = f'Question: {r["problem"].strip()}\nAnswer:'
        return [(ctx, ' ' + o.strip()) for o in r['options']]
    # WinoGrande: the sentence is the first line of `problem`, with `_` marking the blank
    sent = r['problem'].split('\n')[0].strip()
    pre, post = sent.split('_', 1)
    return [(pre + o.strip(), post) for o in r['options']]


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from modeling_echo.torch_weight_loader import build_and_load_torch
    from patch_baseline_mtp import patch_baseline_mtp_weights

    src = sorted(glob.glob(str(D / 'data/plat' / a.bench / '*.jsonl')),
                 key=lambda p: Path(p).stat().st_mtime)[-1]
    items = [json.loads(l) for l in open(src)]
    if a.limit:
        items = items[:a.limit]

    torch.cuda.set_device(a.gpu)
    tok = AutoTokenizer.from_pretrained(str(D / 'tokenizer'), trust_remote_code=True,
                                        local_files_only=True)
    model, cfg, stats = build_and_load_torch(a.ckpt, device=f'cuda:{a.gpu}', dtype=torch.bfloat16)
    patch_baseline_mtp_weights(model, a.ckpt, 24, True)
    model.eval()
    lm = model.lm_head

    def logprob(ctx, cont):
        """sum log P(cont | ctx), and the byte length of cont for acc_norm."""
        c = [2] + tok.encode(ctx, add_special_tokens=False)
        k = tok.encode(cont, add_special_tokens=False)
        ids = c + k
        t = torch.tensor([ids], device=f'cuda:{a.gpu}')
        with torch.no_grad():
            h, _, _ = model.model(t, use_cache=False, return_mtp_hidden=True)
            # position len(c)-1+i predicts k[i]
            s, e = len(c) - 1, len(ids) - 1
            logp = torch.log_softmax(lm(h[:, s:e]).float()[0], dim=-1)
            tgt = torch.tensor(k, device=f'cuda:{a.gpu}')
            lp = logp.gather(1, tgt[:, None])[:, 0].sum().item()
        return lp, max(len(cont.encode('utf-8')), 1)

    t0 = time.time()
    out, hit, hit_norm = [], 0, 0
    for i, r in enumerate(items):
        scored = [logprob(c, k) for c, k in pairs(a.bench, r)]
        gi = r['options'].index(r['gold']) if r['gold'] in r['options'] else -1
        pick = max(range(len(scored)), key=lambda j: scored[j][0])
        pick_n = max(range(len(scored)), key=lambda j: scored[j][0] / scored[j][1])
        hit += pick == gi
        hit_norm += pick_n == gi
        out.append({'id': r['id'], 'gold_idx': gi, 'pick': pick, 'pick_norm': pick_n,
                    'logp': [s[0] for s in scored], 'bytes': [s[1] for s in scored]})
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(items)} acc {100 * hit / (i + 1):.2f} '
                  f'acc_norm {100 * hit_norm / (i + 1):.2f} {time.time() - t0:.0f}s', flush=True)

    n = sum(1 for x in out if x['gold_idx'] >= 0)      # items whose gold text matches an option
    res = {'bench': a.bench, 'cfg': a.cfg, 'ckpt': a.ckpt, 'protocol': 'lm-eval loglikelihood',
           'n': len(out), 'n_scorable': n,
           'acc': 100 * hit / max(n, 1), 'acc_norm': 100 * hit_norm / max(n, 1),
           'pick_rate_first_option': 100 * sum(x['pick'] == 0 for x in out) / len(out),
           'seconds': round(time.time() - t0, 1), 'items': out}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, 'w'))
    print(f'{a.bench} {a.cfg}: acc {res["acc"]:.2f} acc_norm {res["acc_norm"]:.2f} '
          f'(n={n}, 选第一项 {res["pick_rate_first_option"]:.1f}%) -> {OUT}')


if __name__ == '__main__':
    main()
