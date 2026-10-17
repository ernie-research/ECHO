"""echo target 的 2x2 温度格 x 任务 entropy 横轴，用**当前最新**数据重画。

版式照 `workspace/env_run/figs/grid_target_draft_accept.png`：四格 = target x draft
温度组合，横轴 = target predictive entropy，marker = c1/4/8/16 均值，误差棒 = 四档 std。

和那张旧图的区别（都是数据侧的）：
  * 第一格用 §7.1 重测的 `cell1b_*`（09-22 23:43），不再是 09-21 21:23 改动之前的数据；
  * 第二、四格的 EAGLE3-cold 用 `ce3rs_tgt0/tgt1`、warm2 用 `wf2c_t1rs/t1rsk`
    —— 即 `SGLANG_EAGLE3_OFFICIAL_RS=0` 的 classic actual-q + d2t 散射链。旧图那两格
    标注的 "EAGLE3 omitted: rejected_probs=zeros" 已经不成立；
  * 第二、三格的 DSpark 用 `SGLANG_DSPARK_DRAFT_MODE` 强制解耦的
    `dsgreedytgt_sampled` / `dssampletgt_greedy`（§5：它们是真实测量，不是退化重复）；
  * 补上 DFlash-b8 四格（`dflash_t0` / `dflash_t0_draft1` / `dflash_t1_draft0` /
    `dflash_t1rs`）和 EAGLE3-warm2 ep2。
  * DSpark-b16 照旧不画（预算 17 slot，和其余 8 slot 的 arm 不同量级）。

baseline：优先用 arm 自己 run 里的；`t1rs_/t1k_/t1rsk_` 那批是按 arm 拆 run 的
（03:40 / 11:12 同批启动，baseline 单独一个 run），这种情况用同格内同批的 baseline。

    python3 _plot_grid_entropy.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = os.environ.get("ECHO_RUNS", str(HERE.parent / "h800_c1_runs/specbench_runs"))
ENTROPY = os.environ.get("ECHO_ENTROPY", str(HERE / "entropy_summary_echo1.5.json"))
OUT = os.environ.get("ECHO_FIGS", os.path.dirname(os.path.abspath(__file__)))
CONCS = (1, 4, 8, 16)

# 每格按优先级列 run（前面的先占位，后面的不覆盖已有 arm）。修后的链排在前面。
CELLS = {
    "c1": ["*__cell1b_mtp", "*__cell1b_e3ds", "*__dflash_t0",
           "*__dsown_nospec_t0"],
    "c2": ["*__ce3rs_tgt0", "*__dsgreedytgt_sampled", "*__dflash_t0_draft1",
           "*__dsown_nospec_c2", "*t1rsMERGED*", "*__t1rs_echo_mtp_k3",
           "*__t1rs_baseline"],
    "c3": ["*__dssampletgt_greedy", "*__dflash_t1_draft0",
           "*__dsown_nospec_c3", "*t1kMERGED*"],
    "c4": ["*__ce3rs_tgt1", "*__dflash_t1rs", "*__dsown_nospec_c4",
           "*t1rskMERGED*"],
}
CELL_TITLE = {
    "c1": "target T=0  /  draft T=0",
    "c2": "target T=0  /  draft T=1",
    "c3": "target T=1  /  draft T=0",
    "c4": "target T=1  /  draft T=1 (RS)",
}
# 热启动那条线要的是 **DS-MTP warm**（ds_mtp1 target，nospec 语料 ep2），不是 echo 的
# warmfixed2。它跑在另一个 target 上，所以 speedup 必须除它自己 run 的 baseline
# （§9.5 的 ALT_BASE 陷阱：按 (ds, conc) 共享分母会把 echo 的 baseline 套上去）。
# accept 的绝对值因此不与 echo 的 arm 直接可比，横轴的熵仍是 echo target 算的。
ARMS = [
    ("echo_mtp", "ECHO-MTP (K=7)", "tab:red", "o", "-"),
    ("eagle3_cold_final", "EAGLE3-cold", "tab:blue", "s", "-"),
    ("eagle3_wdsnospec_ep2", "EAGLE3 warm from DS-MTP (own target)",
     "darkgoldenrod", "P", "-"),
    ("dspark_b7", "DSpark-b7", "tab:green", "^", "-"),
    ("dflash_b8", "DFlash-b8", "tab:brown", "*", "-"),
]
CONC = 1          # 只画并发 1，所以不再有跨并发的均值/误差棒


def load_cell(patterns):
    """res[(arm, ds, conc)] = {'accept', 'speedup'}；先出现的 run 优先，不被覆盖。

    baseline 分两层：**优先用 arm 自己那个 run 里的**（跨 target 的 dsown_* 必须如此，
    否则分母会被 echo 的 baseline 顶掉）；只有 run 里根本没带 baseline 时（t1rs_/t1k_
    那批是按 arm 拆 run 的）才退回同格共享的 baseline。
    """
    shared, per_dir = {}, []
    for pat in patterns:
        for d in sorted(glob.glob(f"{RES}/{pat}")):
            if "INVALID" in d or "PURGED" in d or not os.path.exists(f"{d}/results.jsonl"):
                continue
            base, rows = {}, []
            for line in open(f"{d}/results.jsonl"):
                r = json.loads(line)
                if r["num_samples"] < 4 * r["concurrency"]:      # 铁律
                    continue
                k = (r["dataset"], r["concurrency"])
                if r["method"] == "baseline":
                    base[k] = float(r["extra"]["decode_toks_per_s"])
                else:
                    rows.append((r, k))
            for k, v in base.items():
                shared.setdefault(k, v)
            per_dir.append((base, rows))
    res = {}
    for base, rows in per_dir:
        for r, k in rows:
            key = (r["method"], *k)
            if key in res:
                continue
            b = base.get(k) or shared.get(k)
            res[key] = {"accept": r["accept_length"] or None,
                        "speedup": (float(r["extra"]["decode_toks_per_s"]) / b) if b else None}
    return res


def value(res, arm, ds, metric):
    return (res.get((arm, ds, CONC)) or {}).get(metric)


def panel(ax, ent, res, metric, title):
    order = [d for d in sorted(ent, key=ent.get)
             if any(value(res, a, d, metric) is not None for a, *_ in ARMS)]
    for arm, label, color, marker, ls in ARMS:
        pts = [(ent[d], v) for d in order
               if (v := value(res, arm, d, metric)) is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                color=color, ls=ls, lw=1.5, ms=5.5, label=label)
    if metric == "speedup":
        ax.axhline(1.0, color="k", lw=0.9)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    for d in order:
        ax.annotate(d, (ent[d], 0.02), xycoords=("data", "axes fraction"),
                    rotation=90, fontsize=7, ha="center", va="bottom", color="dimgrey")
    return order


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ent = {d["dataset"]: d["entropy"] for d in json.load(open(ENTROPY))["datasets"]}
    cells = {cell: load_cell(pats) for cell, pats in CELLS.items()}
    for cell, res in cells.items():
        cov = {lab: sum(1 for d in ent if value(res, arm, d, "accept") is not None)
               for arm, lab, *_ in ARMS}
        print(f"{cell}: " + "  ".join(f"{k}={v}" for k, v in cov.items()))
    # Decode speedup is reported in the paper and is not re-rendered here.
    for metric, ylab, fname in (
            ("accept", "accept length\n(tokens per verify step)",
             "grid_target_draft_accept_c1.png"),):
        fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, sharey="row")
        for ax, cell in zip(axes.flat, CELLS):
            panel(ax, ent, cells[cell], metric, CELL_TITLE[cell])
        for ax in axes[1]:
            ax.set_xlabel("target predictive entropy (nats/token)", fontsize=10)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab, fontsize=10)
        axes.flat[0].legend(fontsize=8, loc="lower left", ncol=2)
        fig.suptitle(
            "ECHO 1.5 (a1b-10B MoE, tp8), concurrency 1: "
            f"{'accept length' if metric == 'accept' else 'decode-only speedup'} "
            "vs task entropy across target x draft temperature\n"
            "EAGLE3 cells 2/4 use the fixed RS chain (OFFICIAL_RS=0); "
            "DSpark cells 2/3 are force-decoupled.\n"
            "DS-MTP warm-start runs on its own target (ds_mtp1_sft step5000) and its "
            "own baseline; the entropy axis is echo's.",
            fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(f"{OUT}/{fname}", dpi=150)
        print(f"wrote {OUT}/{fname}")


if __name__ == "__main__":
    main()
