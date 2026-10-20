# ECHO figure plotting

All plotting entry points, required inputs, and generated files are separated into three
directories:

```text
figure_plot/
├── scripts/   plotting scripts and shared plotting helpers
├── inputs/    CSV data required by the scripts
└── outputs/   generated PDF and PNG figures
```

The scripts resolve paths relative to their own location, so they can be run from any working
directory. Matplotlib 3.9.2 is the reference version. Every figure here is shipped as the paper typeset it.

## Generate figures

From the repository root:

```bash
python3 figure_plot/scripts/plot_fig1a.py
python3 figure_plot/scripts/plot_fig1b.py
python3 figure_plot/scripts/plot_fig3a.py
python3 figure_plot/scripts/plot_fig3b.py
python3 figure_plot/scripts/plot_fig4.py
python3 figure_plot/scripts/plot_fig5.py
python3 figure_plot/scripts/plot_fig7.py
```

| Script | Outputs |
|---|---|
| `plot_fig1a.py` | `outputs/figure1a.{pdf,png}` |
| `plot_fig1b.py` | `outputs/figure1b.{pdf,png}` |
| `plot_fig3a.py` | `outputs/figure3a.{pdf,png}` |
| `plot_fig3b.py` | `outputs/figure3b.{pdf,png}` |
| `plot_fig4.py` | `outputs/figure4a.{pdf,png}`, `figure4b.{pdf,png}`, `figure4c.{pdf,png}` |
| `plot_fig5.py` | `outputs/figure5.{pdf,png}`, `figure6.{pdf,png}` |
| `plot_fig7.py` | `outputs/figure7.{pdf,png}` |

## Inputs

The benchmark cells any figure draws are the rows of `tab:benchmark_results`,
`tab:train-twice` and `tab:sft`, so they are transcribed from the paper into
`scripts/paper_tables.py` rather than kept as data files. Figures 1a, 3a and 4
read them from there. The remaining inputs are measurements the paper reports
only as figures:

| Input | Used by |
|---|---|
| `trainloss_pair_mainlm.csv` | Figures 5 and 6 |
| `trainloss_families.csv` | Figures 5, 6 and 7 |
| `trainloss_mtp3.csv` | Figures 5, 6 and 7 |
| `intro_al7_points.csv` | Figure 1b |
| `serving_metrics.csv` | Figure 3b |

In `intro_al7_points.csv`, the ECHO-1.5 and ECHO-3 rows come from sweeps that draft to K=8, so every
depth plotted is measured. The other three come from sweeps to K=6; there `den7 = 0` marks `p7` as
extrapolated, continuing the geometric decay of the last two measured rates, `p7 = p6 * min(p6 / p5, 1)`.

`serving_metrics.csv` is a compact export of target entropy, accepted length, and corrected
decode-speedup values formerly reconstructed at plotting time from the raw serving archive.
The derivation scripts and raw measurements are retained under `archive/serving/`, while the
export keeps Figure 3b independent of that larger archive.

Shared colors live in `scripts/figure_palette.py` and `scripts/method_colors.py`.

## Archive

`archive/serving/` holds the two accepted-length serving diagnostics that the paper discusses
but does not print as a numbered figure: the target × draft temperature grid and the
same-temperature diagonal. Both were measured on the paper's serving stack, one node of eight
H800 GPUs, every run at concurrency 1, and both regenerate from the records kept alongside them.
See that directory's README.
