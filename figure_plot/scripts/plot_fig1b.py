#!/usr/bin/env python3
"""Figure 4: accepted length against serving depth, with the legend it was missing.

Redrawn from intro_al7_points.csv (tau_K1..K7 per configuration, the K-sweep
harness export behind the original figure). Style is the ZCR revision: the
red/grey palette shared with Figure 1 and Figure 3, the 3.9in teaser canvas and
margins, one solid line per configuration with a marker at every measured depth (the
K-sweep is discrete): filled inside the trained depth, hollow beyond it, which reads at
this size where dashed and dotted segments did not. Each curve is named at its right end instead of in a legend, so
the x axis runs past K=7 to make room; the axes rectangle keeps its original
position (46.332, 32.076, 206.388, 199.584 measured from the top).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_palette as P
from plot_paths import INPUT_DIR, OUTPUT_DIR

MCOL = {"MTP-1": P.MTP_1, "MTP-3": P.MTP_3, "ECHO-1": P.ECHO_1,
        "ECHO-1.5$^{(3)}$": P.ECHO_15, "ECHO-3": P.ECHO_3}
CSV_LABEL = {"MTP-1": "MTP-1", "MTP-3": "MTP-3", "ECHO-1": "ECHO-1",
             "ECHO-1.5$^{(3)}$": "ECHO-1.5", "ECHO-3": "ECHO-3"}
ORDER = ["MTP-1", "MTP-3", "ECHO-1", "ECHO-1.5$^{(3)}$", "ECHO-3"]
# ECHO-1.5 (5.54) and ECHO-3 (5.61) end within 0.07 of each other, so their names
# are pushed apart vertically; the rest sit on their own endpoint.
LABEL_Y = {"MTP-1": 2.04, "MTP-3": 4.42, "ECHO-1": 2.83,
           "ECHO-1.5$^{(3)}$": 5.42, "ECHO-3": 5.80}

rows = {}
with (INPUT_DIR / "intro_al7_points.csv").open() as f:
    for r in csv.DictReader(f):
        rows[r["label"]] = (int(r["trained_depth"]),
                            [float(r[f"tau_K{k}"]) for k in range(1, 8)])

plt.rcParams.update({"font.size": 7.4, "axes.linewidth": 1.0,
                     "axes.edgecolor": P.AXIS, "text.color": P.AXIS,
                     "axes.labelcolor": P.AXIS, "xtick.color": P.AXIS,
                     "ytick.color": P.AXIS})   # text in AXIS grey, as Figure 1a beside it
# Canvas chosen so that, at \includegraphics[width=2.22in], this panel's axes
# rectangle lands at the same page height as teaser (a)'s: same drawn height and the
# same offset below the subfigure box top. Solved from the two measured files;
# the axes drawing area keeps its original 206.388x199.584 shape.
# Canvas solved so that, at \includegraphics[width=2.48in], this panel's axes
# rectangle matches teaser (a)'s on the page: same drawn height and the same offset
# below the subfigure box top. Re-solve both if either include width changes.
CW, CH = 178.6, 203.7 # native size == the 2.60in the paper includes it at
AX_L, AX_R, AX_B, AX_H = 34.0, 7.0, 50.0, 131.21
fig, ax = plt.subplots(figsize=(CW / 72, CH / 72))
ax.set_position([AX_L / CW, AX_B / CH, (CW - AX_L - AX_R) / CW, AX_H / CH])
ax.grid(alpha=.25)

ax.axvspan(1.5, 7.4, color=P.ZONE, alpha=.09, zorder=0)
ax.axvspan(3.5, 7.4, color=P.ZONE, alpha=.11, zorder=0)
for td, xc, ha in ((1, 2.5, "center"), (3, 5.5, "center")):
    ax.axvline(td + .5, color=P.AXIS, lw=1.3, ls=(0, (5, 2)), zorder=2)
    ax.text(xc, 6.05, f"beyond {td}", ha=ha, va="center",
            fontsize=5.0, color=P.AXIS)

x = list(range(1, 8))
for lab in ORDER:
    td, al = rows[CSV_LABEL[lab]]
    col = MCOL[lab]
    # the K-sweep is measured at integer depths only, so every point carries a marker
    # one line style throughout: at this size dashes were illegible. Extrapolation is
    # carried by the markers instead -- filled inside the trained depth, hollow beyond it.
    ax.plot(x, al, "-", color=col, lw=1.5, zorder=3)
    ax.plot(x[:td], al[:td], "o", ls="none", color=col, ms=4.6, mfc=col, mec=col,
            zorder=4)
    ax.plot(x[td - 1:], al[td - 1:], "o", ls="none", ms=4.6, mfc="white", mec=col,
            mew=1.3, zorder=4)
    ax.plot([x[td - 1]], [al[td - 1]], "o", ls="none", ms=4.6, mfc=col, mec=col,
            zorder=5)
    ax.annotate(lab, (7.35, LABEL_Y[lab]), ha="left", va="center",
                fontsize=6.8, color=col, zorder=5)

ax.set_xticks(x)
ax.set_xlim(.75, 10.5)
ax.set_ylim(1.6, 6.2)
ax.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
ax.set_xlabel("serving depth $K$", fontsize=8.0)
ax.set_ylabel("accepted length", fontsize=8.0)

for ext in ("pdf", "png"):
    fig.savefig(OUTPUT_DIR / f"figure1b.{ext}", dpi=300)
plt.close(fig)
print("outputs/figure1b.{pdf,png}")
for lab in ORDER:
    td, al = rows[CSV_LABEL[lab]]
    print(f"  {lab:14s} td={td}  K3={al[2]:.3f}  K7={al[6]:.3f}")
