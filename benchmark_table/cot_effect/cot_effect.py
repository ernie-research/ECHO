#!/usr/bin/env python3
"""Does the gain over MTP-1 survive taking chain-of-thought away?

The same 216 BBH items, the same few-shot prompt file, decoded two ways: once
letting the model reason before answering, once made to answer directly (the
token budget drops from 512 to 16). Only the answering mode changes, so the two
columns isolate the effect of the chain of thought rather than of a different
prompt.

    python3 cot_effect.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = ["echo_mtp1", "ds_mtp1", "echo_mtp1.5"]
LABEL = {"echo_mtp1": "ECHO-1", "ds_mtp1": "MTP-1", "echo_mtp1.5": "ECHO-1.5"}
MODES = ["direct", "cot"]          # direct is the default reading


def macro(path):
    d = json.load(open(path))
    groups = collections.defaultdict(list)
    for s in d["samples"]:
        groups[s["group"]].append(float(bool(s["correct"])))
    per_group = [sum(v) / len(v) for v in groups.values()]
    return 100.0 * sum(per_group) / len(per_group), d["n"], len(groups), d.get("max_new")


def main():
    score = {}
    print(f"{'answering':<12}{'configuration':<12}{'score':>8}{'items':>8}"
          f"{'sub-tasks':>11}{'token budget':>14}")
    for mode in MODES:
        for cfg in CONFIGS:
            s, n, g, mn = macro(os.path.join(HERE, f"BBH_{mode}__{cfg}.json"))
            score[(mode, cfg)] = s
            print(f"{mode:<12}{LABEL[cfg]:<12}{s:>8.2f}{n:>8}{g:>11}{mn:>14}")

    print(f"\n{'gain over MTP-1':<20}{'answering directly':>20}{'with reasoning':>17}")
    for cfg in ("echo_mtp1", "echo_mtp1.5"):
        direct = score[("direct", cfg)] - score[("direct", "ds_mtp1")]
        cot = score[("cot", cfg)] - score[("cot", "ds_mtp1")]
        print(f"{LABEL[cfg]:<20}{direct:>+20.2f}{cot:>+17.2f}")

    print("""
  ECHO-1.5 is ahead of MTP-1 either way, and further ahead when the reasoning is
  taken away. ECHO-1's gap to MTP-1 is small in both columns and changes sign
  between them, which 216 items cannot resolve: read it as no measurable
  difference rather than as a reversal.

  The reasoning column here (216 items of the CoT prompt) sits close to the
  paper's BBH row, which grades the same prompt over the full 6,511 items in
  ../results/BBH__*.json.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
