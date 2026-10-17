"""Same-temperature cells only: target and draft both T=0, and both T=1 (RS).

`_plot_grid_entropy.py` draws the full 2x2 of target x draft temperature. The two mixed cells
(target 0 / draft 1 and target 1 / draft 0) answer a different question -- what happens when the
draft is sampled at a temperature the target does not use -- and they need the force-decoupled DSpark
runs plus the fixed-RS EAGLE3 chain to be meaningful at all. The two diagonal cells are the ones that
correspond to how anybody actually serves speculative decoding, so this pulls them out side by side:
top-left (T=0/T=0) and bottom-right (T=1/T=1) of that grid.

Everything else is imported from `_plot_grid_entropy` unchanged -- the same run patterns, the same
per-arm baseline rule (an arm's own run first, the shared one only when the run carries none), the
same entropy axis (echo's target), concurrency 1. So a number here is the same number as in the 2x2.

    python3 _plot_grid_same_temp.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "temperature_grid"))
import _plot_grid_entropy as G                                            # noqa: E402
G.OUT = os.environ.get("ECHO_FIGS", str(HERE))                            # noqa: E402

DIAG = ["c1", "c4"]          # target=draft: 0/0 and 1/1(RS)


def main() -> None:
    os.makedirs(G.OUT, exist_ok=True)
    ent = {d["dataset"]: d["entropy"] for d in json.load(open(G.ENTROPY))["datasets"]}
    cells = {c: G.load_cell(G.CELLS[c]) for c in DIAG}
    for c in DIAG:
        cov = {lab: sum(1 for d in ent if G.value(cells[c], arm, d, "accept") is not None)
               for arm, lab, *_ in G.ARMS}
        print(f"{c} ({G.CELL_TITLE[c]}): " + "  ".join(f"{k}={v}" for k, v in cov.items()))
    # Decode speedup is reported in the paper and is not re-rendered here.
    for metric, ylab, fname in (
            ("accept", "accept length\n(tokens per verify step)",
             "diag_same_temp_accept.png"),):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
        for ax, c in zip(axes, DIAG):
            G.panel(ax, ent, cells[c], metric, G.CELL_TITLE[c])
            ax.set_xlabel("target predictive entropy (nats/token)", fontsize=10)
        axes[0].set_ylabel(ylab, fontsize=10)
        axes[0].legend(fontsize=8, loc="lower left", ncol=2)
        fig.suptitle(
            "ECHO 1.5 (a1b-10B MoE, tp8), concurrency 1, target and draft at the SAME "
            f"temperature: {'accept length' if metric == 'accept' else 'decode-only speedup'} "
            "vs task entropy\n"
            "left = both greedy, right = both T=1 with the fixed rejection-sampling chain "
            "(OFFICIAL_RS=0). DS-MTP warm-start runs on its own target and its own baseline; "
            "the entropy axis is echo's.",
            fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(f"{G.OUT}/{fname}", dpi=150)
        print(f"wrote {G.OUT}/{fname}")


if __name__ == "__main__":
    main()
