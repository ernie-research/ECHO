# Serving diagnostics

Two accepted-length diagnostics behind the serving discussion, both on the paper's serving
stack: SGLang on one node of eight H800 GPUs, tensor parallelism 8, FlashAttention-3. Every
run here is at concurrency 1 (`dp1xc1`); the paper reports no concurrency sweep and neither
does this directory.

- `h800_c1_runs/specbench_runs/`: the per-run SGLang records the two diagnostics reduce.
- `temperature_grid/`: accepted length over the target × draft temperature grid, with the
  target entropy per dataset it is keyed on. `_plot_grid_entropy.py` regenerates it.
- `same_temperature/`: accepted length along the same-temperature diagonal, with the points
  the figure plots and the run identifiers behind them. `_plot_grid_same_temp.py` regenerates
  it.

Decode speedup is reported in the paper and nowhere else here, so neither script renders it.
