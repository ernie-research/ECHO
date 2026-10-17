#!/usr/bin/env python3
"""Teaser (a): the two stages side by side, each against the baseline of its stage.

The stages get their own axes on purpose. Every bar is a gain over the DeepSeek
baseline *of its own stage*, and those are two different models, so a shared zero
line would read as if the finetuned ECHO were being compared with the pretrained
MTP-1. Each panel names its baseline in its title.

One symbol per configuration: colour alone identifies a configuration and repeats
across the panels, so the legend names configurations and nothing else. Grey stays
with the MTP family and the red ramp deepens with ECHO's prediction depth, the
palette of Figure 1b. Bars are laid out on one absolute grid of unit pitch and the
panel widths follow the bar counts, so a bar is the same width in both panels.

Cells: paper_tables.STEP78K (the rows of tab:benchmark_results) for the pretraining
panel and paper_tables.SFT (the rows of tab:sft) for the other.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figure_palette as P
from paper_tables import SFT as sft_rows, STEP78K as rows
from plot_paths import OUTPUT_DIR

RC = ["GSM8K", "MATH", "BBH", "BBEH", "PIQA", "WinoGrande",
      "HumanEval+", "MBPP+", "LiveCodeBench"]
QA = ["MMLU", "MMLU-Pro", "PreciseWikiQA", "HotPotQA", "PoPQA"]

# configuration -> (table column, colour); the palette of Figure 1b
COL = {"MTP-3": ("deepseek_mtp3", P.MTP_3),
       "ECHO-1": ("echo_mtp1", P.ECHO_1),
       r"ECHO-1.5$^{(3)}$": ("echo_mtp1.5", P.ECHO_15),
       "ECHO-3": ("echo_mtp3", P.ECHO_3)}
PRE = ["MTP-3", "ECHO-1", r"ECHO-1.5$^{(3)}$", "ECHO-3"]
POST = [r"ECHO-1.5$^{(3)}$"]
GRID = P.GRID
CLASSES = [("reasoning\n& code\n9/14", RC), ("QA & multi.\nchoice\n5/14", QA)]
SHORT = ["reas. &\ncode\n9/14", "QA &\nmulti.\n5/14"]
W = 0.8                      # bar width, in the unit pitch shared by both panels


def pre(col, ks):
    return sum(float(rows[b][col]) - float(rows[b]["deepseek_mtp1"]) for b in ks) / len(ks)


def post(ks):
    return sum(float(sft_rows[b]["echo_mtp1.5"]) - float(sft_rows[b]["deepseek_mtp1"])
               for b in ks) / len(ks)


def slots(order):
    """bar centres on the unit grid: one slot per bar, one empty slot between classes"""
    n = len(order)
    out, x = [], 0.0
    for g in range(len(CLASSES)):
        out.append([x + i for i in range(n)])
        x += n + 1.0
    return out


# drawn at the width the paper includes it at (subfigure 3.60in), so the include
# scale is 1: tight-bbox margins are not magnified and the three-line class labels
# cannot be clipped by the caption
FIG_W, FIG_H = 3.60, 2.72
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                     "axes.edgecolor": P.INK, "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": P.INK,
                     "ytick.color": P.INK, "xtick.labelcolor": "black",
                     "ytick.labelcolor": "black"})        # all text black; axes stay INK
def xspan(order):
    """the x range a panel will occupy, in grid units; width_ratios must follow this
    (not the bar count) or the unit pitch, and with it the drawn bar width, differs"""
    grid = slots(order)
    return (grid[-1][-1] + 0.5 + W / 2) - (-0.5 - W / 2)


span = [xspan(PRE), xspan(POST)]
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True,
                         gridspec_kw={"width_ratios": span, "wspace": 0.10})

for ax, order, value, title in (
        (axes[0], PRE, lambda c, ks: pre(COL[c][0], ks), "pretraining"),
        (axes[1], POST, lambda c, ks: post(ks), "after SFT")):
    grid = slots(order)
    for g, (name, ks) in enumerate(CLASSES):
        for xi, conf in zip(grid[g], order):
            v = value(conf, ks)
            ax.bar([xi], [v], W, color=COL[conf][1], edgecolor=P.INK,
                   lw=0.4, zorder=3)
            ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 1.5 if v >= 0 else -7.5), ha="center",
                        fontsize=6.6, zorder=4)
    ax.set_title(title, fontsize=7.8, color="black", pad=3)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color=P.INK, lw=0.9, zorder=4)
    ax.set_xticks([sum(s) / len(s) for s in grid])
    names = [n for n, _ in CLASSES] if len(order) > 1 else SHORT
    ax.set_xticklabels(names, fontsize=7.3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_xlim(-0.5 - W / 2, grid[-1][-1] + 0.5 + W / 2)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel(r"$\Delta$ benchmark vs MTP-1", fontsize=8.0)
axes[0].set_ylim(-0.50, 3.02)

# legend as in Figure 3: the baseline is a line, every entry carries its 14-task average
AVG = {"MTP-1": 35.22, "MTP-3": 35.38, "ECHO-1": 35.73,
       r"ECHO-1.5$^{(3)}$": 36.34, "ECHO-3": 36.36}
ORDER = ["MTP-1"] + PRE
# matplotlib fills a legend column-major, so the names go in the first column and the
# scores in the second; padding the labels with spaces cannot align them, the glyphs
# have different widths
swatch = {"MTP-1": Line2D([], [], color=P.MTP_1, lw=1.0)}
swatch.update({c: Patch(facecolor=COL[c][1], edgecolor=P.INK, lw=0.4) for c in PRE})
blank = lambda: Line2D([], [], color="none")
# four legend columns -- name, score, name, score -- so five entries take three rows
# instead of five; matplotlib fills column-major, hence the interleaving below
# two rows, three name/score pairs per row: the legend spans the panel instead of
# stacking, which keeps it out of the bars. matplotlib fills column-major, so each
# name column is followed by its score column.
nrow = 2
cells = [(c, f"{AVG[c]:.2f}") for c in PRE]
groups = [cells[i:i + nrow] for i in range(0, len(cells), nrow)]
handles, labels = [], []
for col in groups:
    handles += [swatch[c] if c else blank() for c, _ in col]
    labels += [c if c else "" for c, _ in col]
    handles += [blank() for _ in col]
    labels += [v if v else "" for _, v in col]
leg = axes[0].legend(handles, labels, title="14-task average (MTP-1: 35.22)",
                     fontsize=6.0, title_fontsize=6.0, loc="upper left", ncol=4,
                     handlelength=0.9, handletextpad=0.28, columnspacing=0.42,
                     labelspacing=0.18, borderpad=0.28, framealpha=1.0)
leg._legend_box.align = "left"
leg.get_frame().set_edgecolor(P.MTP_3)
leg.get_frame().set_linewidth(0.6)

# the right panel names its own baseline and configuration, same convention
# The panel is 45pt wide, too narrow for a legend box, so the two 14-task averages are
# set as a small two-column block: names flush left, scores flush right, which aligns
# both columns without relying on glyph widths. The panel title carries the stage, so
# the names need no suffix.
for row, (name, score) in enumerate(((r"MTP-1", "36.27"),
                                     (r"ECHO-1.5$^{(3)}$", "37.88"))):
    y = 0.905 - row * 0.068
    axes[1].text(0.04, y, name, transform=axes[1].transAxes, ha="left",
                 va="top", fontsize=5.4, color="black")
    axes[1].text(0.97, y, score, transform=axes[1].transAxes, ha="right",
                 va="top", fontsize=5.4, color="black")
axes[1].text(0.04, 0.99, "14-task average", transform=axes[1].transAxes,
             ha="left", va="top", fontsize=5.4, color="black")
fig.subplots_adjust(left=0.135, right=0.978, top=0.885, bottom=0.215)
for ext in ("pdf", "png"):
    fig.savefig(OUTPUT_DIR / f"figure1a.{ext}", dpi=300)
print("outputs/figure1a.{pdf,png}")
for name, ks in CLASSES:
    n = name.replace("\n", " ")
    pres = "  ".join(f"{c.replace('$^{(3)}$','(3)')} {pre(COL[c][0], ks):+.2f}" for c in PRE)
    print(f"  {n:22s} pretrain: {pres}   |  after SFT: {post(ks):+.2f}")
