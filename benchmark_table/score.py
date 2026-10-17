#!/usr/bin/env python3
"""Re-derive the table cells from the per-item records in `results/`.

No score is read from a file. For every benchmark that is graded by a rule, the
cell is recomputed here from the per-item records and compared with the value
the paper reports. The two code benchmarks are graded by executing the generated
programs, so they are not recomputed here; see the note this script prints.

    python3 score.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CONFIGS = ["echo_mtp1", "ds_mtp1", "echo_mtp1.5"]

# the values the paper reports, for comparison only; nothing is read from them
PAPER = {
    "GSM8K":      {"echo_mtp1": 61.49, "ds_mtp1": 61.64, "echo_mtp1.5": 63.31},
    "BBH":        {"echo_mtp1": 46.44, "ds_mtp1": 46.91, "echo_mtp1.5": 48.55},
    "MMLU":       {"echo_mtp1": 55.93, "ds_mtp1": 56.39, "echo_mtp1.5": 56.93},
    "MMLU-Pro":   {"echo_mtp1": 26.02, "ds_mtp1": 26.60, "echo_mtp1.5": 27.58},
    "PIQA":       {"echo_mtp1": 77.57, "ds_mtp1": 77.68, "echo_mtp1.5": 78.38},
    "WinoGrande": {"echo_mtp1": 63.85, "ds_mtp1": 62.27, "echo_mtp1.5": 64.40},
}
BY_EXECUTION = {
    "HumanEval+":    {"echo_mtp1": 26.22, "ds_mtp1": 23.17, "echo_mtp1.5": 29.27},
    "LiveCodeBench": {"echo_mtp1": 7.00, "ds_mtp1": 6.00, "echo_mtp1.5": 7.00},
}


def recompute(path):
    """Unweighted macro-average over sub-tasks, from the per-item records."""
    d = json.load(open(path))

    # length-normalised log-likelihood over the candidate continuations.
    # Items the harness could not score carry gold_idx = -1 and are excluded,
    # which is what the protocol's n_scorable counts.
    if "items" in d:
        items = [i for i in d["items"] if int(i["gold_idx"]) >= 0]
        hit = sum(1 for i in items if int(i["pick_norm"]) == int(i["gold_idx"]))
        return 100.0 * hit / len(items), len(items), 1

    # generated answers: per-item correctness averaged inside each sub-task
    # first, then across sub-tasks
    groups = collections.defaultdict(list)
    for s in d["samples"]:
        groups[s["group"]].append(float(bool(s["correct"])))
    per_group = [sum(v) / len(v) for v in groups.values()]
    return 100.0 * sum(per_group) / len(per_group), len(d["samples"]), len(groups)


def main():
    print(f"{'benchmark':<14}{'configuration':<14}{'recomputed':>11}{'paper':>8}"
          f"{'items':>8}{'sub-tasks':>11}")
    bad = 0
    for bench, paper in PAPER.items():
        for cfg in CONFIGS:
            path = os.path.join(RESULTS, f"{bench}__{cfg}.json")
            got, n, g = recompute(path)
            ok = abs(got - paper[cfg]) < 0.005
            bad += not ok
            print(f"{bench:<14}{cfg:<14}{got:>11.2f}{paper[cfg]:>8.2f}{n:>8}{g:>11}"
                  + ("" if ok else "   <-- MISMATCH"))
    total = len(PAPER) * len(CONFIGS)
    print(f"\n{total - bad} of {total} rule-graded cells match the paper")

    print("\nScored by execution, not recomputed here:")
    for bench, paper in BY_EXECUTION.items():
        for cfg in CONFIGS:
            d = json.load(open(os.path.join(RESULTS, f"{bench}__{cfg}.json")))
            print(f"  {bench:<14}{cfg:<14}paper {paper[cfg]:>6.2f}   "
                  f"{len(d['samples'])} generations in the file")
    print("""
  The generated program of every item is in the `output` field. To reproduce
  these two, run `scoring/code_score.py` (HumanEval+, against the EvalPlus
  tests) and `scoring/lcb_score.py` (LiveCodeBench) over those outputs; both
  execute the programs, which is why the numbers are not derived here.""")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
