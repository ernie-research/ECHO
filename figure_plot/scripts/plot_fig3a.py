#!/usr/bin/env python3
"""Figure 3(a): gain over MTP-1 per task category, pretraining, legend above the axes.

Each bar is the unweighted mean over the category's benchmarks of (config - DeepSeek MTP-1), from
paper_tables.STEP78K (the rows of tab:benchmark_results); the legend carries the 14-task
averages. Colours from figure_palette.py. The canvas is the printed subfigure width, so the font
sizes below are the sizes on the page.

    python3 scripts/plot_fig3a.py      # -> outputs/figure3a.{pdf,png}
"""
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402
from matplotlib.lines import Line2D                                           # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle                     # noqa: E402
from matplotlib.transforms import Affine2D                                   # noqa: E402

import figure_palette as P                                                    # noqa: E402
from paper_tables import STEP78K as rows                                      # noqa: E402
from plot_paths import OUTPUT_DIR                                             # noqa: E402

CATS = {   # tick label -> benchmarks, in plotting order
    "math":          ["GSM8K", "MATH"],
    "reasoning":     ["BBH", "BBEH", "PIQA", "WinoGrande"],
    "code":          ["HumanEval+", "MBPP+", "LiveCodeBench"],
    "mult. choice":  ["MMLU", "MMLU-Pro"],
    "QA":            ["PreciseWikiQA", "HotPotQA", "PoPQA"],
}
BASE = ("MTP-1", "deepseek_mtp1")
SERIES = [  # label, table column; bar order within a category
    ("MTP-3", "deepseek_mtp3"),
    ("ECHO-1", "echo_mtp1"),
    (r"ECHO-1.5$^{(3)}$", "echo_mtp1.5"),
    ("ECHO-3", "echo_mtp3"),
]
BOLD = "ECHO-3"                # legend entry set in bold

W, H = 3.07, 2.9               # inches, as printed (\begin{subfigure}{3.07in}); fixed, not tight
MARGIN = dict(left=0.15, right=0.99, bottom=0.13, top=0.815)   # = plot_fig3b.py
LEG_GAP = 6 / (H * 72)      # legend bottom sits 6pt above the axes, as in plot_fig3b.py
FS = dict(label=8.5, tick=7.5, cat=7.5, legend=6.8)
plt.rcParams.update({"font.size": FS["tick"], "pdf.fonttype": 42})

n_total = sum(len(v) for v in CATS.values())
assert n_total == len(rows) == 14


def delta(benchmarks, col):
    return st.mean(float(rows[b][col]) - float(rows[b][BASE[1]]) for b in benchmarks)


def overall(col):
    return st.mean(float(r[col]) for r in rows.values())


fig, ax = plt.subplots(figsize=(W, H), dpi=300)   # lay out at the saved dpi: the legend values
                                                    # are placed from measured positions
fig.subplots_adjust(**MARGIN)
w = 0.2
for j, (lab, col) in enumerate(SERIES):
    for i, bs in enumerate(CATS.values()):
        x = i + (j - (len(SERIES) - 1) / 2) * w
        ax.bar(x, delta(bs, col), w * 0.92, color=P.CONF[lab], edgecolor="#222222", lw=0.5,
               zorder=3)
ax.axhline(0, color="k", lw=0.8, zorder=4)
ax.grid(axis="y", color=P.GRID, lw=0.6, zorder=0)
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels([f"{c}\n{len(bs)}/{n_total}" for c, bs in CATS.items()],
                   fontsize=FS["cat"], linespacing=1.0, va="top")
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", labelsize=FS["tick"], length=2.5)
ax.set_xlim(-0.55, len(CATS) - 0.45)
ax.set_ylim(-1.2, 3.1)
ax.set_yticks(range(-1, 4))
ax.set_ylabel(r"$\Delta$ benchmark vs MTP-1", fontsize=FS["label"], labelpad=2)
for s in ax.spines.values():
    s.set_linewidth(0.6)

# Legend drawn as a small table: swatch, name, and the 14-task average in its own column, so the
# averages of one column start at the same x (ECHO-1 and ECHO-1.5^(3) share one) and every
# row's name and value sit on one baseline. Laid out in points from the box's lower-left corner;
# the box is centred above the axes, LEG_GAP below it, and as tall as figure3b's legend.
entries = [(l, overall(c)) for l, c in [BASE] + SERIES]
COLUMNS = [[0, 1], [2, 3], [4]]               # indices into entries, top row first
BOX_H, PAD, ROW = 21.1, 3.5, 8.5              # pt: box height, inner padding, row pitch
SW_W, SW_H, SW_GAP, VAL_GAP, COL_GAP = 9.0, 5.5, 2.7, 3.5, 7.0

pt = fig.dpi / 72
r = fig.canvas.get_renderer()


def width(txt, **kw):
    t = fig.text(0, 0, txt, fontsize=FS["legend"], **kw)
    w = t.get_window_extent(r).width / pt
    t.remove()
    return w


def weight(name):
    return "bold" if name == BOLD else "normal"


col_w, x = [], PAD
layout = []                                   # (entry index, x of swatch, x of name, x of value)
for members in COLUMNS:
    name_w = max(width(entries[i][0], fontweight=weight(entries[i][0])) for i in members)
    val_w = max(width(f"{entries[i][1]:.2f}", fontweight=weight(entries[i][0])) for i in members)
    for i in members:
        layout.append((i, x, x + SW_W + SW_GAP, x + SW_W + SW_GAP + name_w + VAL_GAP))
    x += SW_W + SW_GAP + name_w + VAL_GAP + val_w + COL_GAP
box_w = x - COL_GAP + PAD

x0 = (W * 72 - box_w) / 2                     # box origin in points from the figure corner
y0 = (MARGIN["top"] + LEG_GAP) * H * 72
T = Affine2D().translate(x0, y0).scale(1 / 72) + fig.dpi_scale_trans
row_y = {0: BOX_H / 2 + ROW / 2, 1: BOX_H / 2 - ROW / 2}
fig.add_artist(FancyBboxPatch((0, 0), box_w, BOX_H, boxstyle="round,pad=0,rounding_size=2",
                              transform=T, fc="white", ec="0.6", lw=0.6))
for i, xs, xn, xv in layout:
    name, val = entries[i]
    row = next(c.index(i) for c in COLUMNS if i in c)
    y = row_y[row]
    if name == BASE[0]:                       # the baseline is the zero line, so a line
        fig.add_artist(Line2D([xs, xs + SW_W], [y, y], color="k", lw=1.0, transform=T))
    else:
        col = P.CONF[name]
        fig.add_artist(Rectangle((xs, y - SW_H / 2), SW_W, SW_H, transform=T, fc=col,
                                 ec="#222222", lw=0.5))
    for xt, txt in ((xn, name), (xv, f"{val:.2f}")):
        fig.text(xt, y, txt, transform=T, fontsize=FS["legend"], fontweight=weight(name),
                 ha="left", va="center_baseline")

for e in ("pdf", "png"):
    fig.savefig(OUTPUT_DIR / f"figure3a.{e}", dpi=300)
print("figure3a: 14-task averages  " + "  ".join(f"{l} {overall(c):.2f}" for l, c in [BASE] + SERIES))
for c, bs in CATS.items():
    print(f"  {c.replace(chr(10), ' '):14s}" + "  ".join(f"{l} {delta(bs, k):+.2f}" for l, k in SERIES))

# the category labels sit close at this width: report the tightest gap between neighbours
r = fig.canvas.get_renderer()
boxes = [t.get_window_extent(r) for t in ax.get_xticklabels()]
gap = min(b.x0 - a.x1 for a, b in zip(boxes, boxes[1:])) * 72 / fig.dpi
print(f"  tightest gap between category labels: {gap:.1f} pt")
