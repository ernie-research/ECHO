#!/usr/bin/env python3
"""Figure 5 (ECHO-1 against MTP-1) and Figure 6 (all five configurations).

每张都是两栏：左 = 主干 next-token loss，右 = depth-1 的 MTP loss。右栏只画 depth 1，
因为五个配置都训 depth 1，是唯一跨配置可比的 MTP 量（ECHO-1.5 的 depth 2/3 只在被采样
到的 step 上有值）。

y 轴上界按"曲线已经走稳之后"的值定（默认 step CLIP 之后的最大平滑值），训练最初那段冲高
直接跑出画面——这正是把后段差异看清楚的办法。下界取平滑曲线最小值再留一点余量。

数据：
  * ECHO-1 / MTP-1 来自 trainloss_mtp_full.csv（curve=echo/ds，只有 train/loss 与
    train/mtp_1_loss），主干 loss 用 total - 0.3*mtp_1 反推，这个恒等式在
    plot_echo_vs_ds.py 里验过。
  * ECHO-1.5 / ECHO-3 / MTP-3 来自 trainloss_families.csv，直接有 train/main_lm_loss。

用法: python3 plot_loss_families.py [max_step] [ema] [clip_step]
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_palette as P
from plot_paths import INPUT_DIR, OUTPUT_DIR

CSV_PAIR = INPUT_DIR / "trainloss_pair_mainlm.csv"
CSV_FAM = [INPUT_DIR / "trainloss_families.csv", INPUT_DIR / "trainloss_mtp3.csv"]

MAX_STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 78000
EMA_W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.999
CLIP = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
MTP_COEF = 0.3
RAW_EVERY = 20
PAPER = (6.3, 2.4)
FS = {"lab": 6.8, "tick": 6.3, "leg": 6.8}


def ema(values, weight):
    out, last, num = [], 0.0, 0
    for v in values:
        last = last * weight + (1 - weight) * v
        num += 1
        out.append(last / (1 - weight ** num))
    return out


def load(paths):
    d = {}
    for path in ([paths] if isinstance(paths, (str, os.PathLike)) else paths):
        if not os.path.exists(path):
            print(f"  [note] 缺 {path}")
            continue
        for r in csv.DictReader(open(path)):
            d.setdefault((r["curve"], r["tag"]), {})[int(r["step"])] = float(r["value"])
    return d


PAIR, FAM = load(CSV_PAIR), load(CSV_FAM)


def pair_series(curve):
    L, M = PAIR[(curve, "train/loss")], PAIR[(curve, "train/mtp_1_loss")]
    grid = sorted(s for s in set(L) & set(M) if s <= MAX_STEP)
    return grid, [L[s] - MTP_COEF * M[s] for s in grid], [M[s] for s in grid]


def fam_series(curve):
    B, M = FAM[(curve, "train/main_lm_loss")], FAM[(curve, "train/mtp_1_loss")]
    grid = sorted(s for s in set(B) & set(M) if s <= MAX_STEP)
    return grid, [B[s] for s in grid], [M[s] for s in grid]


def build(spec):
    """spec: [(label, colour, loader, curve)] -> [(label, colour, grid, bb, mtp, sm_bb, sm_mtp)]"""
    out = []
    for label, colour, loader, curve in spec:
        try:
            grid, bb, mtp = loader(curve)
        except KeyError as e:
            print(f"  [skip] {label}: missing {e}")
            continue
        if not grid:
            print(f"  [skip] {label}: no common steps")
            continue
        out.append((label, colour, grid, bb, mtp, ema(bb, EMA_W), ema(mtp, EMA_W)))
        print(f"  {label:<10} n={len(grid):<6} step {grid[0]}..{grid[-1]}  "
              f"bb[end]={ema(bb, EMA_W)[-1]:.4f}  mtp1[end]={ema(mtp, EMA_W)[-1]:.4f}")
    return out


def ylim(curves, idx, lo_step, hi_step):
    """只看 [lo_step, hi_step] 窗口内的平滑值定上下界：上界就此压低，开头冲高那段跑出画面。"""
    his, los = [], []
    for c in curves:
        grid, sm = c[2], c[5 + idx]
        win = [v for s, v in zip(grid, sm) if lo_step <= s <= hi_step]
        if not win:
            win = sm
        his.append(max(win))
        los.append(min(win))
    span = max(his) - min(los)
    return min(los) - 0.06 * span, max(his) + 0.08 * span


def draw(curves, out_pdf, out_png, xlo=0, xhi=None, clip=None,
         ylabs=("backbone loss", "MTP loss"), raw=True, raw_every=60, raw_alpha=0.10):
    """raw=False 时不画原始点：y 轴一收紧，per-step 噪声（±0.1）就比窗口（~0.02）大一个量级，
    画出来是一片竖条纹，什么也读不出来。"""
    xhi = MAX_STEP if xhi is None else xhi
    ylo_from = clip if clip is not None else xlo
    for size, fs, out, dpi in ((PAPER, FS, out_pdf, 150),
                               ((11, 4.2), {"lab": 11, "tick": 10, "leg": 10}, out_png, 130)):
        fig, axes = plt.subplots(1, 2, figsize=size)
        for idx, (ax, ylab) in enumerate(zip(axes, ylabs)):
            for label, colour, grid, bb, mtp, sm_bb, sm_mtp in curves:
                series, sm = (bb, sm_bb) if idx == 0 else (mtp, sm_mtp)
                keep = [i for i, s in enumerate(grid) if xlo <= s <= xhi]
                gw = [grid[i] for i in keep]
                if raw:
                    ax.plot(gw[::raw_every], [series[i] for i in keep][::raw_every],
                            color=colour, alpha=raw_alpha, linewidth=0.4)
                ax.plot(gw, [sm[i] for i in keep], color=colour, linewidth=1.5, label=label)
            ax.set_xlabel("pretraining step", fontsize=fs["lab"])
            ax.set_ylabel(ylab, fontsize=fs["lab"])
            ax.tick_params(labelsize=fs["tick"])
            ax.legend(loc="upper right", fontsize=fs["leg"], frameon=False)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(*ylim(curves, idx, ylo_from, xhi))
        fig.tight_layout(pad=0.4)
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  -> {out}")


def audit_fam():
    """每条 fam 曲线：训了几个深度、以及 total = main_lm + 0.3*mean(mtp_k) 是否成立。

    深度数是从数据反解的——框架对没训的深度也会 log tag，但值是 0/缺失，所以用"最后 1k
    步里非零"来判。恒等式用来确认 main_lm_loss 就是主干 next-token loss，没被别的口径污染。
    """
    curves = sorted({c for (c, _) in FAM})
    for c in curves:
        have = [k for k in (1, 2, 3) if (c, f"train/mtp_{k}_loss") in FAM]
        L = FAM.get((c, "train/loss"), {})
        B = FAM.get((c, "train/main_lm_loss"), {})
        if not L or not B:
            print(f"  {c:<7} 缺 train/loss 或 main_lm_loss，跳过审计")
            continue
        steps = sorted(set(L) & set(B))
        tail = [s for s in steps if s > max(steps) - 1000] or steps[-1000:]
        depth, means = [], {}
        for k in have:
            M = FAM[(c, f"train/mtp_{k}_loss")]
            vals = [M[s] for s in tail if s in M]
            means[k] = sum(vals) / len(vals) if vals else 0.0
            if means[k] > 1e-6:
                depth.append(k)
        err = []
        for s in tail:
            ms = [FAM[(c, f"train/mtp_{k}_loss")].get(s) for k in depth]
            ms = [m for m in ms if m is not None]
            if ms:
                err.append(L[s] - (B[s] + MTP_COEF * sum(ms) / len(ms)))
        bb = sum(B[s] for s in tail) / len(tail)
        print(f"  {c:<7} depths={depth} mtp_means="
              f"{ {k: round(v, 4) for k, v in means.items()} } "
              f"main_lm[tail]={bb:.4f} "
              f"identity_err={sum(err) / len(err):+.5f}" if err else
              f"  {c:<7} depths={depth} main_lm[tail]={bb:.4f}")


def common_window(curves, warmup=1000):
    """所有曲线都有数据的 step 区间，起点再推后 warmup 步。

    推后是为了甩掉 EMA 的预热段：ECHO-1.5 从 60001 才有数据，它的 EMA 在起点处没有历史，
    前几百步贴着原始噪声走，画出来是个假尖峰。
    """
    lo = max(c[2][0] for c in curves) + warmup
    hi = min(c[2][-1] for c in curves)
    return lo, hi


FAMLAB = ("backbone loss", "depth-1 MTP loss")

print("--- fam 数据审计 (depths / 恒等式)")
audit_fam()
print(f"MAX_STEP={MAX_STEP} EMA={EMA_W} clip={CLIP}\n--- figure 5 / 6")
# 两家族合成一张：backbone(左) / depth-1(右) 各放全部五条配置。
# 正文那段本来就是跨五条配置论证的（"Across all five it spans 0.006"），
# 分两张图画同样的两个 panel 反而要读者在图间来回对照。
print("--- 两家族合图 (五条配置)")
both = build([("MTP-1", P.MTP_1, pair_series, "ds"),
              ("MTP-3", P.MTP_3, fam_series, "mtp3"),
              ("ECHO-1", P.ECHO_1, pair_series, "echo"),
              ("ECHO-1.5$^{(3)}$", P.ECHO_15, fam_series, "echo15"),
              ("ECHO-3", P.ECHO_3, fam_series, "echo3")])
if both:
    lo, hi = common_window(both)
    print(f"  公共窗口 {lo}..{hi}  ({len(both)} 条曲线)")
    draw(both, OUTPUT_DIR / "figure6.pdf", OUTPUT_DIR / "figure6.png",
         xlo=lo, xhi=hi, ylabs=FAMLAB, raw=False)

print("--- ECHO-1 vs MTP-1 (已在论文里，全程图，只收紧 y 轴)")
pair = build([("ECHO-1", "#c63a4a", pair_series, "echo"),
              ("MTP-1", "#6b7280", pair_series, "ds")])
if pair:
    draw(pair, OUTPUT_DIR / "figure5.pdf", OUTPUT_DIR / "figure5.png",
         xlo=0, xhi=MAX_STEP, clip=CLIP, raw=False)
