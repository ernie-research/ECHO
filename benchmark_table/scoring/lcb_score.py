"""LiveCodeBench graded locally by execution: stdin/stdout for atcoder+codeforces, call-based for
leetcode.

Both styles run in a fresh `python -c` subprocess with a wall-clock timeout, so an infinite loop in a
generation costs one timeout instead of the run. A problem counts as solved only if EVERY kept test
passes (LiveCodeBench pass@1, all tests).

Style details that matter for correctness of the grader itself:
  * stdin: the generated program is run as-is with the test input on stdin; comparison is per-line with
    trailing whitespace stripped, which is what the upstream harness does (a trailing newline or a
    padded line must not fail a correct program).
  * call: the generated code defines `class Solution`; the test input is one JSON value PER LINE, each
    line being one argument, and the expected output is a single JSON value. The driver picks the only
    public method of Solution, so it does not need the method name from starter_code.
    LeetCode problems whose answer is a list-of-lists in arbitrary order are NOT special-cased -- the
    upstream harness does not either.

Self-test: `python eval/lcb_score.py --selftest` runs three hand-written correct solutions (one stdin,
one call-based, one deliberately wrong) and asserts the grader says pass/pass/fail. Run it before
trusting any number out of this file -- a silently broken executor would score every column 0.

Usage:
  python eval/lcb_score.py --selftest
  python eval/lcb_score.py --pred results/lcb/<bench>__<cfg>.json [--set-macro] [--jobs 32]
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(__file__).resolve().parents[1]
REC = D / 'data' / 'official' / 'lcb' / 'records.jsonl'
FENCE = re.compile(r'```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)', re.S)

CALL_DRIVER = '''
import json, sys, inspect
_args = [json.loads(l) for l in sys.stdin.read().split("\\n") if l.strip() != ""]
_m = [f for n, f in inspect.getmembers(Solution, predicate=inspect.isfunction)
      if not n.startswith("_")]
assert len(_m) >= 1, "no public method on Solution"
_r = _m[0](Solution(), *_args)
print(json.dumps(_r))
'''


def extract(text):
    """The program inside the first fenced block; fall back to the raw text when unfenced."""
    m = FENCE.search(text or '')
    code = m.group(1) if m else (text or '')
    return code.strip()


def run(src, stdin, timeout):
    try:
        p = subprocess.run([sys.executable, '-c', src], input=stdin, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return p.stdout if p.returncode == 0 else None


def same_lines(got, want):
    if got is None:
        return False
    g = [l.rstrip() for l in got.rstrip().split('\n')]
    w = [l.rstrip() for l in want.rstrip().split('\n')]
    return g == w


def same_json(got, want):
    if got is None:
        return False
    try:
        return json.loads(got.strip()) == json.loads(want.strip())
    except Exception:
        return got.strip() == want.strip()


def solved(rec, text, timeout=8):
    code = extract(text)
    if not code:
        return False, 'no-code'
    if rec['style'] == 'call':
        if 'class Solution' not in code:
            return False, 'no-Solution'
        src = code + '\n' + CALL_DRIVER
        for t in rec['tests']:
            if not same_json(run(src, t['input'], timeout), t['output']):
                return False, 'wrong'
        return True, 'ok'
    for t in rec['tests']:
        if not same_lines(run(code, t['input'], timeout), t['output']):
            return False, 'wrong'
    return True, 'ok'


SELFTEST_STDIN = '''```python
import sys
data = sys.stdin.read().split()
n = int(data[0])
for s in data[1:1+n]:
    diff = sum(1 for a, b in zip(s, "abc") if a != b)
    print("YES" if diff in (0, 2) else "NO")
```'''
SELFTEST_CALL = '''```python
class Solution:
    def countSeniors(self, details: list) -> int:
        return sum(1 for d in details if int(d[11:13]) > 60)
```'''


def selftest():
    recs = {r['id']: r for r in (json.loads(l) for l in open(REC))}
    call = recs['2727']                                    # leetcode countSeniors
    ok, why = solved(call, SELFTEST_CALL)
    print(f'call-based  2727  -> {ok} ({why})')
    assert ok, 'the call-based driver rejects a hand-written correct solution'
    ok_bad, why_bad = solved(call, '```python\nclass Solution:\n    def f(self, d):\n        return 0\n```')
    print(f'call-based  wrong -> {ok_bad} ({why_bad})')
    assert not ok_bad, 'the grader accepts a wrong solution'
    ok_no, why_no = solved(call, 'no code here at all')
    print(f'call-based  empty -> {ok_no} ({why_no})')
    assert not ok_no
    stdin_rec = next(r for r in recs.values() if r['id'] == '1873_A')
    ok_s, why_s = solved(stdin_rec, SELFTEST_STDIN)
    print(f'stdin       1873_A -> {ok_s} ({why_s})')
    assert ok_s, 'the stdin driver rejects a hand-written correct solution'
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--pred')
    ap.add_argument('--set-macro', action='store_true')
    ap.add_argument('--jobs', type=int, default=32)
    ap.add_argument('--timeout', type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    recs = {r['id']: r for r in (json.loads(l) for l in open(REC))}
    d = json.load(open(a.pred))
    by = {str(s['id']): s.get('output') or '' for s in d['samples']}
    ids = [i for i in recs if i in by]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda i: solved(recs[i], by[i], a.timeout), ids))
    ok = sum(r[0] for r in res)
    why = {}
    for i, (o, w) in zip(ids, res):
        why[w] = why.get(w, 0) + 1
    macro = 100.0 * ok / max(len(ids), 1)
    print(f'{Path(a.pred).name:56s} n={len(ids):4d}  pass@1 {macro:6.2f}  {why}')
    if a.set_macro:
        d['macro'] = macro
        d['macro_source'] = 'lcb_score.py'
        json.dump(d, open(a.pred, 'w'))
        print(f'    wrote macro={macro:.2f} into {a.pred}')


if __name__ == '__main__':
    main()
