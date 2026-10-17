"""Code benchmarks graded locally, against the official test suites, and audited on the platform's
own generations.

Per-item platform verdicts come from the result archives (`output.tar.gz/*/records.jsonl`, extracted
to data/plat/<sub>/records.jsonl); micro pass@1 over those reproduces every reported code score
exactly (HumanEval+ 32.32, MBPP+ 47.35, CruxEval-input 7.50, CruxEval-output 14.02).

Tests come from the official releases, with one honest gap: the platform used the EvalPlus suites,
whose release tarballs are not reachable from here, so HumanEval is graded with the original 164
`check()` harnesses and MBPP with the 3-7 sanitized asserts. That is a *weaker* bar than "+", so
the audit below reports how far our verdict lands from the platform's per item; the same suites grade
every checkpoint, so cross-checkpoint differences remain meaningful.

CruxEval is graded the way the benchmark defines it, by execution rather than string match: for
output prediction the extracted literal must equal f(input), for input prediction the extracted
argument must make the stored assertion true.

Usage:
  python eval/code_score.py                                  # audit platform generations
  python eval/code_score.py --pred results/platfull/humaneval-plus-base__echo_mtp1.json
"""
import argparse, gzip, json, re, subprocess, sys, tempfile, time
from pathlib import Path

D = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument('--bench', default='all',
                choices=['all', 'humaneval-plus-base', 'mbppplus-base', 'cruxeval-input-base',
                         'cruxeval-output-base'])
ap.add_argument('--pred', default='', help='eval_plat_full.py output to grade instead of plat_text')
ap.add_argument('--timeout', type=float, default=10.0)
ap.add_argument('--jobs', type=int, default=16)
ap.add_argument('--show', type=int, default=0)
ap.add_argument('--set-macro', action='store_true',
                help='write our pass@1 back into --pred as its "macro". eval_plat_full.py leaves '
                     'macro at 0.0 on the code sets because it cannot execute tests, so the summary '
                     'tables skip those cells until this runs.')
a = ap.parse_args()

# a completion has stopped being the function body once it starts something new at column 0
STOP = re.compile(r'\n(?=(?:def |class |if __name__|print\(|@|#|assert |"""|\'\'\'|from |import ))')
FENCE = re.compile(r'```(?:python|py)?\s*\n(.*?)(?:```|\Z)', re.S)
ANSWER = re.compile(r'\[ANSWER\](.*?)(?:\[/ANSWER\]|\Z)', re.S)
ASSERT_EQ = re.compile(r'assert\s+f\((.*)\)\s*==\s*(.*?)\s*$', re.S)


def body(text):
    """Function body a completion-style prompt expects (HumanEval)."""
    return STOP.split(text or '', maxsplit=1)[0]


END_MARK = re.compile(r'\n?\[(?:END|DONE)\]')


def whole_program(text):
    """Full solution a task-style prompt expects (MBPP): fenced block if present, else from the
    first def/import onwards."""
    # The MBPP template closes the solution with [END]; handing that to python is a SyntaxError, which
    # is why a probe whose 8/8 outputs were correct solutions still scored 0.00 until this cut existed.
    text = END_MARK.split(text or '', maxsplit=1)[0]
    m = FENCE.search(text or '')
    if m and 'def ' in m.group(1):
        return m.group(1)
    lines = (text or '').split('\n')
    i = next((k for k, l in enumerate(lines) if l.startswith(('def ', 'import ', 'from '))), None)
    return '\n'.join(lines[i:]) if i is not None else (text or '')


def crux_literal(text, want='output'):
    """The literal CruxEval asks for, from the model's `[ANSWER] assert f(...) == X` block."""
    seg = ANSWER.search(text or '')
    seg = seg.group(1) if seg else (text or '')
    m = None
    for line in seg.split('\n'):
        m = ASSERT_EQ.search(line.strip()) or m
        if m:
            break
    if not m:
        return None
    return (m.group(1) if want == 'input' else m.group(2)).strip()


def run(src, timeout):
    """Model-written code goes into a throwaway subprocess, never into this interpreter.

    Returns True (tests passed), False (tests failed) or None (no verdict). The third state is what
    makes --jobs 16 safe. Two things that are not wrong answers used to be counted as wrong ones:
    a fork that the box refuses (BlockingIOError from _fork_exec once sixteen threads each want a
    process) and a script that outlasts the budget (every EvalPlus block imports numpy and some
    carry 790 kB of test inputs). MBPP+ read 7.67 against the platform's 47.35 that way, while
    --jobs 1 reproduced 47.35 exactly. Both cases now retry, then fall through to None.
    """
    for budget in (timeout, timeout * 3):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=True) as f:
            f.write(src)
            f.flush()
            try:
                p = subprocess.run([sys.executable, f.name], timeout=budget, capture_output=True,
                                   env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0'})
                return p.returncode == 0
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                time.sleep(1.0)
                continue
    return None


def program(sub, item, text):
    """The script whose exit status decides pass/fail."""
    if sub == 'humaneval-plus-base':
        t = HE[item['id']]
        # Two output shapes have to be graded here. A completion model continues the prompt, so the
        # script is prompt + body. A chat-v2 turn restates the whole solution inside a ```python fence
        # (all 164 outputs of all four *_sft cells do), so prompt + body would paste a fence in the
        # middle of a signature and every item would fail on a SyntaxError -- which is exactly how
        # those four columns scored 0.00. The fence is the discriminator on purpose: a column-0 `def`
        # is not, since 71 of the platform's own 164 completions have one and they still want the
        # completion path.
        whole = whole_program(text)
        if FENCE.search(text or '') and f'def {t["entry_point"]}' in whole:
            return whole + '\n\n' + t['test'] + f'\ncheck({t["entry_point"]})\n'
        return t['prompt'] + body(text) + '\n\n' + t['test'] + f'\ncheck({t["entry_point"]})\n'
    if sub == 'mbppplus-base':
        t = MBPP.get(str(item['id']))
        if t is None:
            return None
        if PLUS:
            return (whole_program(text) + '\n\n' + t['test'] + '\n')
        return (whole_program(text) + '\n\n' + '\n'.join(t.get('test_imports') or []) + '\n'
                + '\n'.join(t['test_list']) + '\n')
    t = CRUX[item['id']]
    if sub == 'cruxeval-output-base':
        lit = crux_literal(text, 'output')
        if lit is None:
            return None
        return f'{t["code"]}\nassert f({t["input"]}) == ({lit})\n'
    lit = crux_literal(text, 'input')
    if lit is None:
        return None
    return f'{t["code"]}\nassert f({lit}) == ({t["output"]})\n'


# EvalPlus suites, pulled as the HuggingFace parquet export (the GitHub release tarballs 404 from
# here) -- same 164 and 378 items the platform graded, so this is the platform's own bar, not the
# weaker original one. The base suites stay as a fallback if the plus files are absent.
HEP = D / 'data/official/humanevalplus.jsonl'
MBP = D / 'data/official/mbppplus.jsonl'
PLUS = HEP.exists() and MBP.exists()
HE = {r['task_id']: r for r in (json.loads(l) for l in
                                open(HEP if PLUS else D / 'data/official/humaneval_tests.jsonl'))}
MBPP = ({str(r['task_id']): r for r in (json.loads(l) for l in open(MBP))} if PLUS else
        {str(r['task_id']): r for r in json.load(open(D / 'data/official/sanitized-mbpp.json'))})
CRUX = {r['id']: r for r in (json.loads(l) for l in open(D / 'data/official/cruxeval.jsonl'))}
SUBS = (['humaneval-plus-base', 'mbppplus-base', 'cruxeval-input-base', 'cruxeval-output-base']
        if a.bench == 'all' else [a.bench])

for sub in SUBS:
    items = [json.loads(l) for l in open(D / 'data/plat' / sub / 'records.jsonl')]
    if a.pred:
        by = {str(s['id']): s['output'] for s in json.load(open(a.pred))['samples']}
        texts = [by.get(str(it['id']), '') for it in items]
    else:
        texts = [it['plat_text'] for it in items]
    from concurrent.futures import ThreadPoolExecutor
    srcs = [program(sub, it, t) for it, t in zip(items, texts)]
    with ThreadPoolExecutor(a.jobs) as ex:
        verdicts = list(ex.map(lambda s: run(s, a.timeout) if s else False, srcs))
    # a non-verdict is re-run alone: 16 test scripts at once is what makes them miss the budget
    stuck = [k for k, v in enumerate(verdicts) if v is None]
    for k in stuck:
        verdicts[k] = run(srcs[k], max(a.timeout * 6, 60))
    oks = [v is True for v in verdicts]
    ungraded = sum(v is None for v in verdicts)
    n = len(items)
    ours = sum(oks)
    plat = sum(bool(it['plat_score']) for it in items)
    agree = sum(o == bool(it['plat_score']) for o, it in zip(oks, items))
    noans = sum(s is None for s in srcs)
    print(f'{sub:22s} n={n:4d}  ours {100*ours/n:6.2f}  platform {100*plat/n:6.2f}  '
          f'agree {100*agree/n:6.2f}%  no-answer-extracted {noans:4d}'
          + (f'  retried {len(stuck)}' if stuck else '')
          + (f'  UNGRADED {ungraded}' if ungraded else ''))
    if a.pred and a.set_macro:
        if ungraded:
            print(f'    refusing to write macro: {ungraded} items never produced a verdict, so the '
                  f'cell would read low for a reason that is not the model. Re-run with --jobs 1.')
            continue
        d = json.load(open(a.pred))
        d['macro'] = 100.0 * ours / n
        d['macro_source'] = 'code_score.py'
        json.dump(d, open(a.pred, 'w'))
        print(f'    wrote macro={d["macro"]:.2f} into {a.pred}')
    shown = 0
    for it, t, o, s in zip(items, texts, oks, srcs):
        if o != bool(it['plat_score']) and shown < a.show:
            print(f'    [{it["id"]}] plat={it["plat_score"]} ours={o} src={None if s is None else repr(s[-160:])}')
            shown += 1
