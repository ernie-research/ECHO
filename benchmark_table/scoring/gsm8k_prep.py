"""GSM8K (1319 test items) as a local row, using the platform's own MGSM-English prompt.

Why this row exists: the math bucket was carrying a claim on MGSM, and MGSM is literally GSM8K --
Shi et al. took 250 GSM8K test problems and translated them into 10 languages. Its English split IS a
250-item GSM8K subset. Two things go wrong with using it as the math evidence: our local run capped
generation at 640 tokens and truncated 2497/2750 items, and the platform's own per-language numbers are
internally inconsistent (Spanish 11.6 with 1510-character generations against 3000-6300 elsewhere).
GSM8K full set, English only, rule-graded on the `####` gold, is the standard row that removes both
問題: no multilingual stop interactions, 5x the items, and a number comparable with published work.

Prompt: the EXACT 8-shot block the platform serves for MGSM-en (instruction + 8 worked demonstrations,
then the bare question, no trailing cue), so this row and MGSM-en differ only in the item list. That
also means the known quirk carries over: the final question has no "Step-by-Step Answer:" cue, so the
model writes the cue itself and usually opens with a blank line -- `plat_score.score_math` left-strips
before applying the platform's `\\n\\n` stop, which is what made MGSM-en readable at all.

Usage: python eval/gsm8k_prep.py
Writes data/plat/gsm8k-8shot/records.jsonl
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
SRC = D / 'data' / 'official' / 'gsm8k.jsonl'
MGSM = D / 'data' / 'plat' / 'MGSM_wocot_8shot' / 'mgsm_lang_groups.jsonl'
OUT = D / 'data' / 'plat' / 'gsm8k-8shot'


def shot_block():
    """Everything in the MGSM-en prompt before the item's own question."""
    for line in open(MGSM):
        r = json.loads(line)
        if r['group'] != 'en':
            continue
        p, q = r['prompt'], r['problem'].strip()
        i = p.rfind(q)
        assert i > 0, 'the item question is not at the tail of its own prompt'
        return p[:i]
    raise SystemExit('no English MGSM item found')


def main():
    head = shot_block()
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'records.jsonl', 'w') as fh:
        for i, r in enumerate(rows):
            q = (r.get('question') or '').strip()
            fh.write(json.dumps({
                'id': f'gsm8k_{i}', 'subject': None, 'group': 'gsm8k', 'eval_type': 'rule',
                'judger': '', 'problem': q, 'gold': str(r['gold']).strip(), 'options': [],
                # 提示词直接拼进 prompt（与 eval/mgsm_cue_prep.py 一致）。原来靠 --prompt-suffix 传
                # "\nStep-by-Step Answer:"，里面有空格，在 shell 里被词分割成两个参数，argparse 直接
                # 报 unrecognized arguments: Answer:，12 个 cell 全部秒退（2026-09-27 踩到）。
                'prompt': head + q + '\nStep-by-Step Answer:', 'max_tokens': 512, 'stop': ['\n\n'],
                'temperature': 1e-06, 'top_p': 1e-06,
                'plat_text': '', 'plat_score': 0,
            }, ensure_ascii=False) + '\n')
    print(f'wrote {OUT}/records.jsonl  items={len(rows)}  shot_block={len(head)} chars')
    print('  head tail:', repr(head[-90:]))


if __name__ == '__main__':
    main()
