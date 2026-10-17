#!/usr/bin/env python3
"""Figure 4: deeper sampling at both stages, and the lead against run-to-run spread.

Three panels on one shared y axis, each a gain over the matched DeepSeek baseline:

  (a) deeper sampling, pretraining -- ECHO-1.5^(3) against ECHO-2^(7), both vs MTP-1
  (b) deeper sampling, after SFT   -- the same pair, vs MTP-1-Instruct
  (c) ECHO-1 under three conditions -- the baseline run at step 78k, an independent
      repeat of the same recipe, and the baseline run carried to step 156k

Panel (c) is what makes the lead falsifiable: two runs of each model give four
cross-run pairings, all of them in ECHO's favour, and the grey band behind the bars
is the run-to-run standard deviation of the gap estimated from those two runs. The
reasoning & code bars clear it by 3x; QA & multiple choice sits inside it, which is
the same reading the pretraining table gives.

Colour is the configuration, as everywhere else in the paper; in (c) there is one
configuration, so the condition is named under each bar.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import figure_palette as P
import paper_tables
from plot_paths import OUTPUT_DIR

RC = ["GSM8K", "MATH", "BBH", "BBEH", "PIQA", "WinoGrande",
      "HumanEval+", "MBPP+", "LiveCodeBench"]
QA = ["MMLU", "MMLU-Pro", "PreciseWikiQA", "HotPotQA", "PoPQA"]
CLASSES = [("reasoning\n& code", RC), ("QA & multi.\nchoice", QA)]

pre = paper_tables.STEP78K
twice = paper_tables.TRAIN_TWICE

SFT = {key: {b: v[column] for b, v in paper_tables.SFT.items()}
       for key, column in (("ds", "deepseek_mtp1"), ("e15", "echo_mtp1.5"),
                           ("e27", "echo_mtp2_d7"))}
W = 0.8


def d_pre(col, ks):
    return sum(float(pre[b][col]) - float(pre[b]["deepseek_mtp1"]) for b in ks) / len(ks)


def d_sft(key, ks):
    return sum(SFT[key][b] - SFT["ds"][b] for b in ks) / len(ks)


def d_twice(e, b, ks):
    return sum(float(twice[x][e]) - float(twice[x][b]) for x in ks) / len(ks)


# panel (c): the same recipe was pretrained twice on both sides, which gives two
# matched estimates of the gap at step 78k
RUNS = (("echo_78k", "deepseek_78k"), ("echo_repeat_78k", "deepseek_repeat_78k"))


def spread(ks):
    """mean and sample s.d. of the gap over the two matched runs

    With n=2 the sample s.d. is |a-b|/sqrt(2). Cross-pairing the four combinations of
    one run's ECHO with the other's baseline would halve this, but two of those four
    are not comparisons anyone ran.
    """
    a, b = (d_twice(e, d, ks) for e, d in RUNS)
    return (a + b) / 2, abs(a - b) / 2 ** 0.5


PANELS = [
    ("deeper sampling,\npretraining",
     [(r"ECHO-1.5$^{(3)}$", P.ECHO_15, lambda ks: d_pre("echo_mtp1.5", ks)),
      (r"ECHO-2$^{(7)}$", P.ECHO_3, lambda ks: d_pre("echo_mtp2_d7", ks))]),
    ("deeper sampling,\nafter SFT",
     [(r"ECHO-1.5$^{(3)}$", P.ECHO_15, lambda ks: d_sft("e15", ks)),
      (r"ECHO-2$^{(7)}$", P.ECHO_3, lambda ks: d_sft("e27", ks))]),
    # (c) one configuration at two training budgets; the repeat run (run 2) is not drawn, it
    # only enters through the run-to-run s.d. of the error bars (spread() below)
    ("ECHO-1, two\ntraining budgets",
     [("78k steps", P.ECHO_1, lambda ks: d_twice("echo_78k", "deepseek_78k", ks)),
      ("156k steps", P.ECHO_1, lambda ks: d_twice("echo_156k", "deepseek_156k", ks))]),
]
# (c) is one configuration, so both bars keep ECHO-1's colour (the reds of (a)/(b) would read as
# ECHO-1.5 / ECHO-2); the budget is the hatching, and a darker pink edge (which matplotlib also
# uses for the hatch) keeps the pale fill from washing out next to the other two panels
HATCH = {"156k steps": "////"}
EDGE_C = "#c9636c"


def slots(n):
    """bar centres: n bars per class, a 0.6-slot gap between the two classes"""
    out, x = [], 0.0
    for _ in CLASSES:
        out.append([x + i for i in range(n)])
        x += n + 0.6        # a 0.6-slot gap between classes: more width per bar for its label
    return out


def xspan(n):
    g = slots(n)
    return (g[-1][-1] + 0.5 + W / 2) - (-0.5 - W / 2)


plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "hatch.linewidth": 0.6,
                     "axes.edgecolor": P.INK, "text.color": P.INK,
                     "axes.labelcolor": P.INK, "xtick.color": P.INK,
                     "ytick.color": P.INK})
# One file per panel: the paper wraps them in subfigures so they can be cited as
# (a), (b), (c). The y range and the tick set are shared by hand instead of by
# sharey, and the widths are proportional to the bar counts so a bar is the same
# width in all three.
YLIM, YTICKS = (-0.7, 2.95), [-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
OUT = ["figure4a.pdf", "figure4b.pdf", "figure4c.pdf"]
# One canvas and one axes rectangle for all three: LaTeX scales each PDF to its subfigure
# width, so panels of different sizes come out at different heights. Identical files scale
# identically, which keeps the axes, zero lines and legends level at any equal widths.
FIG_W, FIG_H = 2.2, 2.18        # inches
MARGIN = dict(left=0.17, right=0.985, top=0.97, bottom=0.19)
FS = dict(value=7.0, cls=8.0, tick=8.0, legend=7.0)   # pt; was 6.2 / 6.8 / 7.0 / 6.0

for idx, ((title, bars), out) in enumerate(zip(PANELS, OUT)):
    first, third = idx == 0, idx == 2
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    axes = [ax, ax, ax]          # the band code below indexes axes[2]
    grid = slots(len(bars))
    for g, (_, ks) in enumerate(CLASSES):
        for xi, (lab, col, fn) in zip(grid[g], bars):
            v = fn(ks)
            if third:
                ax.bar([xi], [v], W, color=col, edgecolor=EDGE_C, lw=0.8, zorder=3,
                       hatch=HATCH.get(lab))
            else:
                ax.bar([xi], [v], W, color=col, edgecolor=P.INK, lw=0.4, zorder=3)
            # clear the cap when (c) draws an interval on this bar
            lift = spread(ks)[1] if third else 0.0
            if v >= 0:
                ax.annotate(f"{v:+.2f}", (xi, v + lift), textcoords="offset points",
                            xytext=(0, 2.0), ha="center", fontsize=FS["value"], zorder=6)
            else:
                # below the axis the tick labels are in the way, so negatives are
                # written just above the zero line, clear of the interval's upper cap
                ax.annotate(f"{v:+.2f}", (xi, max(0.0, v + lift)),
                            textcoords="offset points", xytext=(0, 2.0),
                            ha="center", fontsize=FS["value"], zorder=6)
    ax.grid(axis="y", color=P.GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color=P.INK, lw=0.9, zorder=4)
    # one row of task-class labels in every panel; in (c) the budget is in the legend
    ax.set_xticks([sum(s) / len(s) for s in grid])
    ax.set_xticklabels([n for n, _ in CLASSES], fontsize=FS["cls"])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.set_xlim(-0.5 - W / 2, grid[-1][-1] + 0.5 + W / 2)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) only: error bars instead of a band. Two runs of each model give four
    # cross-run estimates of the gap, and their standard deviation -- 0.22 on either
    # class -- is the uncertainty every condition inherits, so each bar carries it.
    # A bar whose interval excludes zero is a lead the training noise cannot explain.
    ebar = None
    if third:
        for g, (_, ks) in enumerate(CLASSES):
            _, sd = spread(ks)
            for xi, (_, _, fn) in zip(grid[g], bars):
                ebar = ax.errorbar([xi], [fn(ks)], yerr=sd, fmt="none", ecolor=P.INK,
                                   elinewidth=0.8, capsize=3.0, capthick=0.8, zorder=5)
                # snap stem and caps to whole device pixels: unsnapped, the antialiasing
                # draws the same 0.8pt line one or two pixels wide depending on where it lands
                for art in (*ebar.lines[1], *ebar.lines[2]):
                    art.set_snap(True)

    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    # every panel keeps its tick labels (each is its own subfigure); the y title is left to
    # the caption, since it did not fit the shared left margin

    # each panel carries only the configurations it draws
    if third:                  # one configuration: the entries are its two budgets
        handles = [Patch(facecolor=col, edgecolor=EDGE_C, lw=0.8, hatch=HATCH.get(lab),
                         label=lab) for lab, col, _ in bars]
    else:
        seen, handles = set(), []
        for lab, col, _ in bars:
            if col in seen:
                continue
            seen.add(col)
            name = {P.ECHO_15: r"ECHO-1.5$^{(3)}$", P.ECHO_3: r"ECHO-2$^{(7)}$"}[col]
            handles.append(Patch(facecolor=col, edgecolor=P.INK, lw=0.4, label=name))
    if third:
        ebar.set_label("run-to-run s.d.")   # the real errorbar: the legend draws a stem with caps
        handles.append(ebar)
    leg = ax.legend(handles=handles, fontsize=FS["legend"],
                    loc="upper left", ncol=2, handlelength=1.0,
                    handletextpad=0.35, columnspacing=0.8, labelspacing=0.2,
                    borderpad=0.3, framealpha=1.0)
    leg.get_frame().set_edgecolor(P.MTP_3)
    leg.get_frame().set_linewidth(0.6)

    # no titles (the subfigure captions name the panels); one top and bottom margin for all
    # three, so the axes line up at the same 2.18in height
    fig.subplots_adjust(**MARGIN)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{out.removesuffix('.pdf')}.{ext}", dpi=300)
    plt.close(fig)
    print(f"{out}  ({FIG_W:.2f} x {FIG_H:.2f}in)")
    for lab, _, fn in bars:
        print(f"    {lab.replace(chr(10), ' '):12s} RC {fn(RC):+.2f}  QA&MC {fn(QA):+.2f}")

for ks, name in ((RC, "reasoning & code"), (QA, "QA & multiple choice")):
    m, sd = spread(ks)
    print(f"  {name:22s} matched-run mean {m:+.2f}  s.d. {sd:.2f}")
