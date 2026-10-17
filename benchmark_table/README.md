# The benchmark table

Per-item records for ECHO-1, MTP-1 and ECHO-1.5, and the grading that turns them
into the table cells. `score.py` recomputes every cell from the records and
compares it with the paper:

```
$ python3 score.py
benchmark     configuration  recomputed   paper   items  sub-tasks
GSM8K         echo_mtp1           61.49   61.49    1319          1
BBH           echo_mtp1           46.44   46.44    6511         27
MMLU          echo_mtp1           55.93   55.93   14042         57
MMLU-Pro      echo_mtp1           26.02   26.02   12032         14
PIQA          echo_mtp1           77.57   77.57    1725          1
WinoGrande    echo_mtp1           63.85   63.85    1267          1
...
18 of 18 rule-graded cells match the paper
```

No number is read from a summary: each cell comes back out of the per-item
records in `results/`, one file per benchmark and configuration.

## What is in a record file

- Generated-answer benchmarks (GSM8K, BBH, MMLU, MMLU-Pro, HumanEval+,
  LiveCodeBench) — `samples[]`, one entry per item: `id`, `group` (the
  sub-task), `gold`, `correct`, the extraction `rule` that was applied,
  `n_tokens`, and for the code benchmarks the generated program in `output`.
- Log-likelihood benchmarks (PIQA, WinoGrande) — `items[]`, one entry per item:
  `gold_idx`, the pick before and after length normalisation, the per-candidate
  `logp` and the candidate byte lengths, so the normalisation can be redone.
  Items the harness could not score carry `gold_idx = -1`.

A cell is the unweighted mean over sub-tasks of the per-item mean, which is why
`accuracy` and the table value differ on the multi-subject benchmarks.

## The two code benchmarks

HumanEval+ and LiveCodeBench are graded by running the generated programs, so
`score.py` does not recompute them: it reports the paper's value and the number
of generations on hand. To check them, run `scoring/code_score.py` against the
EvalPlus tests and `scoring/lcb_score.py` against the LiveCodeBench tests over
the `output` fields.

## One further check

- `cot_effect/` — the same 216 BBH items and the same few-shot prompt, decoded
  once letting the model reason and once made to answer directly (token budget
  512 against 16), so only the answering mode changes. `cot_effect.py`
  recomputes both columns, with the direct one as the default reading: ECHO-1.5
  is ahead of MTP-1 either way and further ahead without the reasoning (+8.33
  answering directly, +3.24 with it), while ECHO-1's gap is within what 216
  items can resolve in both columns (+0.93 and -0.46). The reasoning column sits
  close to the paper's BBH row, which grades the same prompt over all 6,511
  items in `results/`.

## What is not here

**Five rows carry the platform's own number: MATH, MBPP+, PreciseWikiQA,
HotpotQA and PopQA.** Each of those cells is what the evaluation platform's
scoring chain produced over its full set — 4,739, 378, 5,000, 7,405 and 14,256
items. That chain's answer extraction and its judging rules are not published
with the paper, so none of the five can be reproduced exactly from outside it,
and nothing in this tree claims to.

Local per-item files exist for MATH and MBPP+, but they do not measure the same
quantity as the reported cell: they were produced under a different few-shot
prefix, the local answer-extraction rule for MATH does not match the one the
platform applied, and for MBPP+ the local bar is weaker — the platform graded
against the EvalPlus suites, which were not reachable from the machine these
runs were on, so a local pass is over the original `check()` harnesses and the
sanitized asserts instead. Our own runs of the three closed-book sets cover a
600-item sample rather than the full set and are judged by a different chain, so
they are not a second reading of the same quantity either. BBEH's per-item
verdicts were not retained, so that row is not re-derivable here. Rather than
offer numbers that cannot be put next to the paper's, none of them is included.

The grading code shared with the rows this tree does re-derive is still here:
`answer_extract.py` serves MATH and GSM8K, `code_score.py` serves MBPP+ and
HumanEval+. Generating your own outputs, you can grade those rows with it.

## What the eight carry

- **Six re-derived in full** — GSM8K, BBH, MMLU, MMLU-Pro, PIQA, WinoGrande.
  `score.py` recomputes the cell from the per-item records.
- **Two graded by execution** — HumanEval+ and LiveCodeBench. The generations
  are here and the paper's value is in `macro` (43/164, 38/164, 48/164 and
  28/400, 24/400, 28/400), but the per-item verdicts were not retained, so
  `score.py` reports the value instead of recomputing it; `scoring/code_score.py`
  over the `output` fields regrades them.

### What defines each of the eight

Every value in this table is read back out of the record file it labels — `n`,
`n_total`, `max_new`, `prefix`, `order` and `stops` are fields in each
`results/<benchmark>__<configuration>.json` — so a re-run that reports the same
knobs is measuring the same quantity, and one that does not is not.

| benchmark | items scored | graded by | cap | prefix | order | stop, applied at extraction |
|---|---|---|---|---|---|---|
| GSM8K | 1,319 / 1,319 | rule | 512 | eos | platlen | `["\n\n"]` |
| BBH | 6,511 / 6,511 | rule | 601 | none | platlen | `["\n\n"]` |
| MMLU | 14,042 / 14,042 | rule | 2 | eos | file | — |
| MMLU-Pro | 12,032 / 12,032 | rule | 11 | eos | file | `["Question:"]` |
| PIQA | 1,725 / 1,838 | length-normalised log-likelihood | — | — | — | — |
| WinoGrande | 1,267 / 1,267 | length-normalised log-likelihood | — | — | — | — |
| HumanEval+ | 164 / 164 | execution, EvalPlus tests | 512 | bos | platlen | six code boundaries |
| LiveCodeBench | 400 / 400 | execution, LiveCodeBench tests | 768 | eos | platlen | `["\n### Question", …]` |

`items scored / in the pool` differ only on PIQA, where 113 of the 1,838 carry
`gold_idx = -1` because the harness could not score them. The two
log-likelihood rows involve no generation, so cap, prefix, order and stop do not
apply to them.

`scoring/` is the grading code as it ran: `answer_extract.py` and `gsm8k_prep.py`
the extraction, `eval_mc_accuracy.py` the multiple-choice benchmarks,
`eval_mc_loglik.py` the log-likelihood ones, `code_score.py` and `lcb_*.py` the
code benchmarks. `score.py` recomputes every rule-graded cell from `results/`
and reads nothing else.

One note on reading it. `code_score.py` and `lcb_score.py` grade by executing the
generated programs against the official EvalPlus and LiveCodeBench suites, which
are not redistributed here, and their audit paths read the platform's own
per-item verdicts, which are not published either. Only the merged records in
`results/` are here, so those two paths do not run from this tree as it stands.
