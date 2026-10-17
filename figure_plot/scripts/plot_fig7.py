#!/usr/bin/env python3
"""Figure 7: the depth-2 MTP loss against the optimizer step.

按相同主模型 step 比才是预算对齐的比较：ECHO-1.5 和 ECHO-3 在同一步上的 MTP 开销是 1.5 比
3.0，深层头收到多少次更新是方法本身的选择，不该归一化掉。ECHO-1.5 的 depth-2 只在被采样的
步上有记录（39.5%），所以点更稀、更噪，但不偏——depth-1 上做过的选择效应检验是 +0.0002
(se 0.0004)。

用法: python3 plot_depth_by_step.py [depths...]   默认 2
输出: outputs/figure7.{pdf,png}
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_palette as P
from plot_paths import INPUT_DIR, OUTPUT_DIR

EMA_W = 0.999
WIN = (61001, 78000)
DEPTHS = [int(a) for a in sys.argv[1:]] or [2]


def ema(values, weight):
    out, last, num = [], 0.0, 0
    for v in values:
        last = last * weight + (1 - weight) * v
        num += 1
        out.append(last / (1 - weight ** num))
    return out


F = defaultdict(dict)
for f in ("trainloss_families.csv", "trainloss_mtp3.csv"):
    p = INPUT_DIR / f
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            F[(r["curve"], r["tag"])][int(r["step"])] = float(r["value"])

CURVES = (("ECHO-1.5", "echo15", P.ECHO_15), ("ECHO-3", "echo3", P.ECHO_3))

for k in DEPTHS:
    tag = f"train/mtp_{k}_loss"
    data = {}
    for lab, curve, colour in CURVES:
        M = F[(curve, tag)]
        steps = [s for s in sorted(M) if WIN[0] <= s <= WIN[1] and M[s] > 1e-6]
        if not steps:
            print(f"  [skip] {lab} depth-{k}: 窗口内无数据")
            continue
        vals = [M[s] for s in steps]
        data[lab] = (steps, vals, ema(vals, EMA_W), colour)
        print(f"depth-{k} {lab:<9} n={len(steps):>6} "
              f"({100*len(steps)/(WIN[1]-WIN[0]+1):.1f}% 的步)  末值 {ema(vals, EMA_W)[-1]:.4f}")
    if len(data) < 2:
        continue
    gap = (data["ECHO-1.5"][2][-1] - data["ECHO-3"][2][-1])
    print(f"depth-{k} 末端差 ECHO-1.5 − ECHO-3 = {gap:+.4f}")

    for size, fs, out, dpi in (((3.3, 2.4), {"lab": 6.8, "tick": 6.3, "leg": 6.8},
                                OUTPUT_DIR / "figure7.pdf", 150),
                               ((5.5, 4.0), {"lab": 11, "tick": 10, "leg": 10},
                                OUTPUT_DIR / "figure7.png", 130)):
        fig, ax = plt.subplots(figsize=size)
        his, los = [], []
        for lab, (steps, vals, sm, colour) in data.items():
            ax.plot(steps, sm, color=colour, linewidth=1.5, label=lab)
            his.append(max(sm))
            los.append(min(sm))
        span = max(his) - min(los)
        ax.set_ylim(min(los) - 0.08 * span, max(his) + 0.10 * span)
        ax.set_xlim(*WIN)
        ax.set_xlabel("pretraining step", fontsize=fs["lab"])
        ax.set_ylabel(f"depth-{k} MTP loss", fontsize=fs["lab"])
        ax.tick_params(labelsize=fs["tick"])
        ax.legend(loc="upper right", fontsize=fs["leg"], frameon=False)
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout(pad=0.4)
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  -> {out}")
