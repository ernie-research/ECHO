#!/usr/bin/env python3
"""Figure 3(b): accepted length at T=1 against target entropy, ECHO and the post-hoc drafters.

The compact, derived serving metrics are stored in inputs/serving_metrics.csv, so this plot no
longer needs to traverse the raw serving-run archive.

    python3 scripts/plot_fig3b.py      # -> outputs/figure3b.{pdf,png}
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                               # noqa: E402

import method_colors as MC                                                    # noqa: E402
from plot_paths import OUTPUT_DIR                                             # noqa: E402
from serving_data import ARMS, load_serving_data                              # noqa: E402

ARM_COLOR = {'echo_mtp': MC.BASE['ECHO'],
             'eagle3_cold_final': MC.BASE['EAGLE'],
             'eagle3_wdsnospec_ep2': MC.BASE['MTP warm-start EAGLE']}
MECH = {'echo_mtp': 'ECHO', 'eagle3_cold_final': 'EAGLE3-cold',
        'eagle3_wdsnospec_ep2': 'EAGLE3-warm', 'dspark_b7': 'DSpark',
        'dflash_b8': 'DFlash'}
SHORT = {'humaneval': 'HumanEval', 'mbpp': 'MBPP', 'math500': 'MATH', 'gsm8k': 'GSM8K',
         'livecodebench': 'LCB', 'swe-bench': 'SWE', 'arena-hard': 'Arena',
         'mt-bench': 'MT-Bench', 'summarization': 'Summ.', 'alpaca': 'Alpaca'}
# same canvas margins as plot_fig3a.py, so the two axes boxes line up edge for edge
MARGIN = dict(left=0.15, right=0.99, bottom=0.13, top=0.815)
LEG_GAP = 6 / (2.9 * 72)    # legend bottom 6pt above the axes (figure height 2.9in), as in figure3a
_CACHE = {}


def geom():
    if 'g' not in _CACHE:
        ent, groups, accept, spd = load_serving_data()
        order = sorted(ent, key=ent.get)
        pred = {dataset for dataset, group in groups.items() if group == 'Predictable'}
        pred_hi = max(ent[d] for d in pred)
        open_lo = min(ent[d] for d in ent if d not in pred)
        _CACHE.update(g=(ent, order, (pred_hi + open_lo) / 2), groups=groups,
                      spd=spd, accept=accept)
    return _CACHE['g']


def ds_labels(ax, order, ent, fontsize=8):
    """数据集名竖排在坐标区底部内侧,每个名字一条细引线连回它真实的熵位置。
    低熵簇挤在一起时标签会被推开,引线自然变成斜线。"""
    tr = ax.get_xaxis_transform()
    lo, hi = ax.get_xlim()
    mg = 0.052 * (hi - lo)
    lx = []
    for d in order:
        x = ent[d]
        lx.append(x if not lx else max(x, lx[-1] + mg))
    shift = max(0.0, lx[-1] - ent[order[-1]])
    lx = [v - shift / 2 for v in lx]
    for d, tx in zip(order, lx):
        ax.plot([tx, ent[d]], [0.072, 0.0], transform=tr, color='0.55', lw=.5,
                zorder=4, clip_on=False)
        ax.text(tx, 0.082, SHORT[d], transform=tr, rotation=90, ha='center', va='bottom',
                fontsize=fontsize, color='0.35', zorder=4)


def panel(vals, ylab, ylim, out, ds_font=8, size=(5.4, 3.6), fs=None):
    ent, order, split = geom()
    f = fs or {}
    fig, ax = plt.subplots(figsize=size)
    ax.axvspan(split, max(ent.values()) + .05, color='0.92', zorder=0)
    ax.axvline(split, color='0.55', ls=':', lw=1.1)

    handles = []
    for arm, _lab, color, marker, ls in ARMS:
        color = ARM_COLOR.get(arm, color)
        pts = [(ent[d], vals[(arm, d)]) for d in order if (arm, d) in vals]
        if not pts:
            continue
        ax.plot(*zip(*pts), marker=marker, color=color, ls=ls, lw=1.6, ms=5.5, zorder=3)
        handles.append(plt.Line2D([], [], color=color, marker=marker, ls=ls,
                                  lw=1.6, ms=5.5, label=MECH[arm]))

    lo, hi = min(ent.values()) - .04, max(ent.values()) + .05
    ax.set_xlim(lo, hi)
    ax.set_ylim(*ylim)
    ax.set_xlabel('target entropy (nats/token)', fontsize=f.get('ax', 10))
    ax.set_ylabel(ylab, fontsize=f.get('ax', 10))
    ax.grid(True, alpha=.35, lw=.6)
    ax.tick_params(labelsize=f.get('tick', 9))

    # 两个区域标注左对齐:一个贴左边框内侧,一个贴分界虚线右侧,同一条基线
    pad = 0.012 * (hi - lo)
    # Predictable against the left frame; OpenEnded centred over its shaded region
    for x, t, ha in ((lo + pad, 'Predictable', 'left'), ((split + hi) / 2, 'OpenEnded', 'center')):
        ax.text(x, .965, t, transform=ax.get_xaxis_transform(),
                ha=ha, va='top', color='0.35', zorder=4, fontsize=f.get('reg', 9))

    # legend above the axes, as in figure3a, right-aligned with the axes box; the names keep
    # their line colour so each entry still matches its curve
    leg = fig.legend(handles=handles, loc='lower right', ncol=3,   # flush with the axes' right edge
                     bbox_to_anchor=(MARGIN['right'], MARGIN['top'] + LEG_GAP),
                     fontsize=f.get('leg', 8.5), frameon=True, borderaxespad=0,
                     handlelength=2.0, handletextpad=.4, columnspacing=.9, labelspacing=.25,
                     borderpad=.35, markerscale=.8)
    fr = leg.get_frame()
    fr.set_edgecolor('0.6'); fr.set_linewidth(.6)
    for txt, h in zip(leg.get_texts(), handles):
        txt.set_color(h.get_color())

    ds_labels(ax, order, ent, fontsize=ds_font)
    fig.subplots_adjust(**MARGIN)
    for e in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'{out}.{e}', dpi=200)
    print(f'wrote {out}.pdf (+png)   split at entropy {split:.3f}')
    for arm, *_ in ARMS:
        o = [vals[(arm, d)] for d in ent
             if _CACHE['groups'][d] == 'OpenEnded' and (arm, d) in vals]
        if o:
            print(f'  {MECH[arm]:<12} OpenEnded mean = {sum(o)/len(o):.3f}')


def main():
    geom()
    # drawn at the printed 3.07in subfigure width and the height of figure3a, so the fonts here
    # are the sizes on the page and the two panels sit level
    panel(_CACHE['accept'], 'accepted length (tokens)', (1.0, 5.6), 'figure3b',
          ds_font=6.5, size=(3.07, 2.9), fs=dict(ax=8.5, tick=7.5, leg=6.8, reg=7.5))


if __name__ == '__main__':
    main()
