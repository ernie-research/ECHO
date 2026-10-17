"""LiveCodeBench v1 (code_generation_lite, 400 problems) -> local records + prompts.

Why this exists: LiveCodeBench is the one row of the suite with no platform extraction, so it cannot be
replayed byte-for-byte like the other 26. The platform's own numbers for it are 1.10-2.79 across every
column -- a spread smaller than the code bucket's noise floor (0.39-0.47) -- so the row as delivered
cannot separate configurations. Re-running it locally is therefore a different measurement, not a
reproduction, and the paper has to say so.

What this script does:
  * decodes `private_test_cases` (base64 -> zlib -> pickle -> json for the compressed ones, plain json
    for the rest) and concatenates them with `public_test_cases`
  * splits the two execution styles the benchmark actually contains: 181 leetcode problems are
    call-based (`starter_code` + JSON-encoded args per line), 219 atcoder/codeforces problems are
    stdin/stdout
  * builds the official LiveCodeBench generation prompt (the wording is copied from the upstream
    runner's PromptConstants so the numbers are comparable with published LCB results)
  * caps the number of tests per problem at --max-tests, keeping all public tests first: some problems
    ship 100+ private cases and the grader has to run every one of them per column per problem

Usage: python eval/lcb_prep.py [--max-tests 25]
Writes data/official/lcb/records.jsonl
"""
import argparse
import base64
import json
import pickle
import zlib
from pathlib import Path

D = Path(__file__).resolve().parents[1]
SRC = D / 'data' / 'official' / 'lcb' / 'test.jsonl'
OUT = D / 'data' / 'official' / 'lcb' / 'records.jsonl'

PRE = ('You will be given a question (problem specification) and will generate a correct Python '
       'program that matches the specification and passes all tests. You will NOT return anything '
       'except for the program.\n\n')
# The upstream zero-shot format line ends with a fenced `# YOUR CODE HERE`, which a base (non-instruct)
# checkpoint simply copies: the first smoke run of ds_mtp1 returned "```python\n# YOUR CODE HERE\n```"
# on every item. That is almost certainly why the platform's served LiveCodeBench numbers sit at
# 1.10-2.79 for every column. So the placeholder is gone and two worked demonstrations are prepended --
# one per execution style -- which is the same trick the platform used with its 18 shots, at 2.
FMT_STDIN = ('### Format: Read the inputs from stdin, solve the problem and write the answer to stdout '
             '(do not directly test on the sample inputs). Enclose your code in a ```python block.\n\n')
FMT_CALL = ('### Format: Use the following starter code to write the solution and enclose your code in '
            'a ```python block.\n```python\n{starter}\n```\n\n')
ANSWER = '### Answer: (use the provided format with backticks)\n\n'

SHOT_STDIN = (
    '### Question:\nYou are given two integers a and b on one line. Print their sum.\n\n'
    '### Format: Read the inputs from stdin, solve the problem and write the answer to stdout '
    '(do not directly test on the sample inputs). Enclose your code in a ```python block.\n\n'
    + ANSWER +
    '```python\nimport sys\n\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n```\n\n')
SHOT_CALL = (
    '### Question:\nGiven an integer array nums, return the number of elements that are strictly '
    'greater than the average of the array.\n\n'
    '### Format: Use the following starter code to write the solution and enclose your code in a '
    '```python block.\n```python\nclass Solution:\n    def countAbove(self, nums: List[int]) -> int:\n'
    '        \n```\n\n'
    + ANSWER +
    '```python\nclass Solution:\n    def countAbove(self, nums: List[int]) -> int:\n'
    '        avg = sum(nums) / len(nums)\n        return sum(1 for x in nums if x > avg)\n```\n\n')


def decode_private(s):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode('utf-8')))))


def build_prompt(row, shots):
    call = bool(row['starter_code'].strip())
    fmt = FMT_CALL.format(starter=row['starter_code'].rstrip()) if call else FMT_STDIN
    head = PRE + (SHOT_STDIN + SHOT_CALL if shots else '')
    return (head + '### Question:\n' + row['question_content'].strip() + '\n\n' + fmt + ANSWER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-tests', type=int, default=25)
    ap.add_argument('--shots', type=int, default=2, choices=(0, 2),
                    help='2 = one worked demo per execution style (base checkpoints need it)')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    n_call = n_tests = 0
    with open(OUT, 'w') as fh:
        for row in rows:
            pub = json.loads(row['public_test_cases']) if row['public_test_cases'] else []
            tests = pub + decode_private(row['private_test_cases'])
            kinds = {t['testtype'] for t in tests}
            assert len(kinds) == 1, f"{row['question_id']}: mixed test types {kinds}"
            call = bool(row['starter_code'].strip())
            assert call == (kinds == {'functional'}), \
                f"{row['question_id']}: starter_code={call} but testtype={kinds}"
            tests = tests[:a.max_tests]
            n_call += call
            n_tests += len(tests)
            fh.write(json.dumps({
                'id': row['question_id'], 'platform': row['platform'],
                'difficulty': row['difficulty'], 'title': row['question_title'],
                'style': 'call' if call else 'stdin', 'starter_code': row['starter_code'],
                'prompt': build_prompt(row, a.shots),
                'tests': [{'input': t['input'], 'output': t['output']} for t in tests],
            }, ensure_ascii=False) + '\n')
    print(f'wrote {OUT}  problems={len(rows)}  call-based={n_call}  stdin={len(rows) - n_call}  '
          f'tests={n_tests} (cap {a.max_tests}/problem)  shots={a.shots}')


if __name__ == '__main__':
    main()
