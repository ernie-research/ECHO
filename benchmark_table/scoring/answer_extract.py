"""Answer extraction shared by the HF and framework target-only evals.

Kept in one place so the two frameworks cannot drift: any comparison of their accuracies is only
meaningful if the extraction rules are byte-identical.

Rule order is most-explicit-first, and markers take their FIRST match: with a 256-token budget
these base checkpoints answer the question and then invent further Q&A pairs, so a late
"the answer is X" usually belongs to a fabricated question.
"""
import math
import re

NUM = r'-?\d[\d,]*(?:\.\d+)?'


def to_num(s):
    """Parse a matched number. Models sometimes emit 400-digit runs, which float() turns into inf
    and int(inf) then raises OverflowError mid-run."""
    try:
        v = float(s.replace(',', ''))
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return int(v) if v == int(v) else v


def extract(text):
    """Returns (value, which_rule)."""
    t = text or ''
    m = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', t)   # MATH-style \boxed{...}
    if m:
        v = to_num(re.sub(r'[\s$\\!,]|\\text\{[^}]*\}', '', m[0]))
        if v is not None:
            return v, 'boxed'
    m = re.findall(rf'####\s*({NUM})', t)                      # GSM8K's own marker
    if m:
        return to_num(m[0]), '####'
    m = re.findall(rf'(?:answer|result|total)(?:\s*(?:is|:|=))+\s*\$?({NUM})', t, flags=re.I)
    if m:
        return to_num(m[0]), 'answer_is'
    m = re.findall(rf'"answer"\s*:\s*"?({NUM})"?', t)          # {"answer": "18"}
    if m:
        return to_num(m[0]), 'json'
    m = re.findall(NUM, t)
    if m:
        return to_num(m[-1]), 'last_int'
    return None, 'none'


def first_seg(s):
    """Keep only the answer to the question we asked, dropping self-generated follow-ups."""
    return re.split(r'\n\s*\[?(?:Question|Problem|Q)\]?\s*[:\.]', s or '',
                    maxsplit=1, flags=re.I)[0]


def ref_answer(s):
    return extract(s)[0]


# ── string-preserving variant, for MATH-style answers ────────────────────────────────────────
# `extract` runs every match through to_num, which is wrong for MATH-500: 182/500 golds are LaTeX
# (`\left( 3, \frac{\pi}{2} \right)`, `\text{Evelyn}`, `90^\circ`). to_num returns None for those,
# the rule then falls through to "last number" and `\boxed{\frac{14}{3}}` grades as 3. Use this
# variant whenever the gold answer may not be a number, and compare with `norm_answer`.
def extract_str(text):
    """Returns (raw_string, which_rule) — no numeric coercion."""
    t = text or ''
    m = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', t)
    if m:
        return m[0].strip(), 'boxed'
    m = re.findall(r'####\s*(' + NUM + r')', t)
    if m:
        return m[0], '####'
    m = re.findall(r'(?:answer|result|total)(?:\s*(?:is|:|=))+\s*\$?(' + NUM + r')', t, flags=re.I)
    if m:
        return m[0], 'answer_is'
    m = re.findall(r'"answer"\s*:\s*"?(' + NUM + r')"?', t)
    if m:
        return m[0], 'json'
    m = re.findall(NUM, t)
    if m:
        return m[-1], 'last_number'
    return None, 'none'


def norm_answer(s):
    """Lenient normalisation for string comparison of MATH-style answers."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'\\(?:left|right|!|;|\s)', '', s)
    s = s.replace('\\,', '').replace('\\dfrac', '\\frac').replace('$', '').replace(' ', '')
    s = re.sub(r'^\\text\{(.*)\}$', r'\1', s)
    s = re.sub(r'\.0+$', '', s)
    s = s.replace('^\\circ', '').replace('\\%', '').replace('%', '')
    s = s.rstrip('.')
    # Thousands separators only: a bare comma is meaningful inside LaTeX such as
    # `\left( 3, \frac{\pi}{2} \right)`, so it must not be stripped globally.
    if re.fullmatch(r'-?\d[\d,]*(?:\.\d+)?', s):
        s = s.replace(',', '')
    return s


def ref_answer_str(s):
    """Gold side: GSM8K stores `#### 18`, MATH-500 stores the answer verbatim."""
    m = re.findall(r'####\s*([^\n]+)', s or '')
    if m:
        return m[-1].strip()
    return (s or '').strip()
