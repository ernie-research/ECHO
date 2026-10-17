"""MathBench-style multiple-choice eval: fixed instruction + few-shot demos + letter answer.

Why this exists
---------------
The completion-style GSM8K eval measures "can the model state a number within the budget", and it
ranked echo_mtp3 last (22%) while the user's own benchmark suite ranks echo_mtp3 ABOVE echo_mtp1.
Those two can disagree without any bug, because the tasks differ:

  * completion GSM8K: open-ended generation, needs CoT, answer extracted from free text
  * MathBench-style: fixed instruction, demos that answer DIRECTLY with "The answer is:<letter>",
    single-token decision, format-following dominates

This script reproduces the second protocol (instruction + demo block taken from the user's
MathBench prompt) so both checkpoints are measured on the format their benchmark uses. Questions
come from the local GSM8K set, converted to 4-way multiple choice with DETERMINISTIC distractors
(same options for every checkpoint), because the original benchmark set is not on this machine.

Interpretation: this tests format-following + answer selection, NOT chain-of-thought. A checkpoint
heavily exposed to instruction templates can win here and still lose the completion eval.
"""
import argparse, json, math, re, sys, time
from pathlib import Path

import torch

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D / 'src')); sys.path.insert(0, str(D / 'eval'))
import tvshim  # noqa: E402,F401  broken torchvision in this venv; see eval/tvshim.py
from modeling_echo.torch_weight_loader import build_and_load_torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--dataset', default=str(D / 'data/gsm8k_test_completion_first50.jsonl'))
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--n', type=int, default=50)
ap.add_argument('--max-new', type=int, default=24)
ap.add_argument('--eos', type=int, default=2)
a = ap.parse_args()

INSTRUCTION = (
    'please use the prefix "The answer is:" without any modification, and provide the answer '
    'directly, with no formatting, no bolding, and no markup. For instance: "The answer is: 42" '
    'or "The answer is: yes". If the question is multiple choice with a single correct answer, '
    'the final answer must only be the letter corresponding to the correct answer. For example, '
    '"The answer is: (a)"\n'
)
DEMOS = [
    ("Question:A shop sells notebooks for $3 each. Tom buys 7 notebooks. How much does he pay?\n"
     "A. $21\nB. $24\nC. $18\nD. $10", "A"),
    ("Question:A train travels 300 miles in 5 hours. What is its average speed in miles per hour?\n"
     "A. 50\nB. 60\nC. 75\nD. 65", "B"),
    ("Question:对自然数集合N，下列哪种运算是可结合的，运算定义为a，b∈N____\nA. a*b=a-b\nB. a*b=a+2b\n"
     "C. a*b=min(a，b)\nD. a*b=|a-b|", "C"),
    ("Question:具有7个结点的所有非同构的树有____个。\nA. 7\nB. 11\nC. 12\nD. 14", "B"),
]
PREFIX = INSTRUCTION + ''.join(f'{q}\nThe answer is:{ans}\n\n' for q, ans in DEMOS)

NUM = r'-?\d[\d,]*(?:\.\d+)?'


def ref_answer(gold_text):
    m = re.findall(rf'####\s*({NUM})', gold_text or '')
    if not m:
        return None
    try:
        v = float(m[-1].replace(',', ''))
    except ValueError:
        return None
    return int(v) if math.isfinite(v) and v == int(v) else v


def fmt(v):
    return f'{v:g}'


def distractors(ref):
    """Deterministic wrong options, so every checkpoint sees identical choices."""
    cands = []
    for c in (ref * 2, ref + 10, round(ref / 2, 2) if ref % 2 else ref // 2, ref + 1, ref - 3):
        if c != ref and c not in cands and c > 0:
            cands.append(c)
        if len(cands) == 3:
            break
    while len(cands) < 3:                     # degenerate refs (very small numbers)
        c = ref + 100 + len(cands)
        if c not in cands:
            cands.append(c)
    return cands


def build_question(qtext, ref, idx):
    """Place the correct option at a rotating slot so a constant-letter guesser scores ~25%.

    A first version sorted the options by value; with distractors {2x, x+10, x/2} that put the
    correct answer in slot B on 5/5 sampled questions, i.e. "always answer B" would have scored
    100%. The slot is now idx % 4 and the wrong options fill the rest in ascending order.
    """
    letters = 'ABCD'
    slot = idx % 4
    wrong = sorted(distractors(ref))
    opts = wrong[:slot] + [ref] + wrong[slot:]
    body = '\n'.join(f'{letters[i]}. {fmt(o)}' for i, o in enumerate(opts))
    return f'Question:{qtext.strip()}\n{body}', letters[slot]


def extract_letter(text):
    t = text or ''
    m = re.search(r'answer\s*is\s*:?\s*\(?\s*([A-Da-d])\b', t)
    if m:
        return m.group(1).upper(), 'answer_is'
    m = re.search(r'^\s*\(?\s*([A-Da-d])\s*[).\s]', t)
    if m:
        return m.group(1).upper(), 'leading_letter'
    m = re.search(r'\b([A-D])\b', t)
    if m:
        return m.group(1), 'any_letter'
    return None, 'none'


torch.cuda.set_device(a.gpu)
tok = AutoTokenizer.from_pretrained(str(D / 'tokenizer'), trust_remote_code=True, local_files_only=True)
model, cfg, stats = build_and_load_torch(a.ckpt, device=f'cuda:{a.gpu}', dtype=torch.bfloat16)
model.eval()
lm = model.lm_head

# Scoring is by LIKELIHOOD of the answer letter, not by generation: after the prompt ends with
# "The answer is:" we compare the logits of " A"/" B"/" C"/" D" (each a single token in this
# tokenizer: 1562/1647/1578/1667) and take the argmax. Generation-based scoring made the result
# depend on whether a checkpoint follows the output format at all -- e.g. echo_mtp3 emitted EOS
# immediately on 3/5 smoke-test questions -- which is exactly the confound this eval must avoid.
LETTERS = 'ABCD'
LETTER_IDS = [tok.encode(f' {c}', add_special_tokens=False)[0] for c in LETTERS]
assert len(set(LETTER_IDS)) == 4, LETTER_IDS


@torch.no_grad()
def score_letters(ids):
    t = torch.tensor([ids], device='cuda')
    hidden, _, _ = model.model(t, use_cache=False)
    logits = lm(hidden[:, -1:]).float()[0, -1]
    lp = torch.log_softmax(logits, -1)
    return [float(lp[i]) for i in LETTER_IDS]


rows = [json.loads(x) for x in open(a.dataset) if x.strip()][:a.n]
samples, t0 = [], time.time()
for i, row in enumerate(rows):
    qtext = next(t['value'] for t in row['conversations'] if t['from'] == 'human')
    qtext = re.split(r'\n\s*Answer\s*:', qtext, maxsplit=1)[0]      # drop the completion-style tail
    qtext = re.sub(r'^Question\s*:\s*', '', qtext.strip())
    ref = ref_answer(next(t['value'] for t in row['conversations'] if t['from'] == 'gpt'))
    if ref is None:
        continue
    q, correct = build_question(qtext, ref, len(samples))
    ids = [2] + tok.encode(PREFIX + q + '\nThe answer is:', add_special_tokens=False)
    lps = score_letters(ids)
    pred = LETTERS[max(range(4), key=lambda k: lps[k])]
    margin = sorted(lps)[-1] - sorted(lps)[-2]
    samples.append({'index': i, 'ref_value': ref, 'correct_letter': correct, 'prediction': pred,
                    'logprobs': {LETTERS[k]: round(lps[k], 4) for k in range(4)},
                    'margin': round(margin, 4)})
    ok = sum(1 for s in samples if s['prediction'] == s['correct_letter'])
    print(f"  #{i} gold={correct} pred={pred} margin={margin:.3f} running={ok}/{len(samples)}",
          flush=True)

ok = sum(1 for s in samples if s['prediction'] == s['correct_letter'])
gold_dist, pred_dist = {}, {}
for s in samples:
    gold_dist[s['correct_letter']] = gold_dist.get(s['correct_letter'], 0) + 1
    pred_dist[s['prediction']] = pred_dist.get(s['prediction'], 0) + 1
out = {'ckpt': a.ckpt, 'n': len(samples), 'correct': ok, 'accuracy': ok / max(len(samples), 1),
       'scoring': 'letter_logprob_argmax', 'gold_letter_dist': gold_dist,
       'pred_letter_dist': pred_dist,
       'mean_margin': sum(s['margin'] for s in samples) / max(len(samples), 1),
       'seconds': round(time.time() - t0, 1), 'samples': samples}
json.dump(out, open(a.out, 'w'), indent=2, ensure_ascii=False)
print(f"accuracy={ok}/{len(samples)} ({out['accuracy']*100:.0f}%) gold={gold_dist} "
      f"pred={pred_dist} mean_margin={out['mean_margin']:.3f} time={out['seconds']}s")
