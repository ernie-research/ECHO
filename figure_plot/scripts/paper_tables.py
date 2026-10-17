#!/usr/bin/env python3
"""The three paper tables, transcribed from the paper itself.

Every cell any figure in this repository draws from the benchmark suite is in one of
`tab:benchmark_results`, `tab:train-twice` or `tab:sft`, so the tables are kept here
rather than in a data file: the paper is the record, and a figure that disagrees with
it is a transcription error rather than a second measurement.
"""


def _table(columns, rows):
    return {name: dict(zip(columns, values)) for name, *values in rows}


# tab:benchmark_results -- step 78,000, base models, no EMA
STEP78K = _table(
    ("echo_mtp1", "echo_mtp3", "echo_mtp1.5", "deepseek_mtp1", "deepseek_mtp3",
     "echo_mtp2_d7"),
    [("GSM8K",         61.49, 59.82, 63.31, 61.64, 63.08, 61.33),
     ("MATH",          36.04, 36.01, 34.64, 34.31, 33.78, 33.56),
     ("BBH",           46.44, 50.02, 48.55, 46.91, 40.18, 48.45),
     ("BBEH",           6.63,  6.41,  6.09,  5.33,  6.96,  5.43),
     ("PIQA",          77.57, 78.26, 78.38, 77.68, 77.97, 78.09),
     ("WinoGrande",    63.85, 64.01, 64.40, 62.27, 63.06, 62.59),
     ("MMLU",          55.93, 57.10, 56.93, 56.39, 58.59, 56.95),
     ("MMLU-Pro",      26.02, 26.37, 27.58, 26.60, 29.37, 27.90),
     ("HumanEval+",    26.22, 29.88, 29.27, 23.17, 25.00, 26.83),
     ("MBPP+",         47.62, 48.68, 47.62, 47.62, 47.88, 47.88),
     ("LiveCodeBench",  7.00,  7.00,  7.00,  6.00,  6.00,  5.25),
     ("PreciseWikiQA",  7.88,  8.30,  8.13,  7.48,  7.52,  7.54),
     ("HotPotQA",      21.35, 21.29, 20.65, 21.18, 20.86, 20.57),
     ("PoPQA",         16.23, 15.83, 16.16, 16.44, 15.09, 15.57)])


# tab:train-twice -- the baseline run, an independent second pretraining run of the same
# recipe at the same step, and the baseline run carried to twice its token budget
TRAIN_TWICE = _table(
    ("echo_78k", "deepseek_78k", "echo_repeat_78k", "deepseek_repeat_78k",
     "echo_156k", "deepseek_156k"),
    [("GSM8K",         61.49, 61.64, 60.05, 60.73, 64.82, 65.05),
     ("MATH",          36.04, 34.31, 34.58, 35.37, 38.49, 36.65),
     ("BBH",           46.44, 46.91, 48.15, 47.79, 48.31, 48.79),
     ("BBEH",           6.63,  5.33,  5.83,  5.59,  6.18,  5.29),
     ("PIQA",          77.57, 77.68, 78.67, 76.64, 79.71, 79.59),
     ("WinoGrande",    63.85, 62.27, 64.96, 62.27, 65.67, 63.30),
     ("MMLU",          55.93, 56.39, 56.08, 56.30, 59.23, 59.36),
     ("MMLU-Pro",      26.02, 26.60, 27.04, 24.21, 26.23, 27.28),
     ("HumanEval+",    26.22, 23.17, 28.05, 25.00, 29.88, 26.83),
     ("MBPP+",         47.62, 47.62, 49.21, 48.68, 50.00, 50.26),
     ("LiveCodeBench",  7.00,  6.00,  6.75,  5.25,  7.25,  6.50),
     ("PreciseWikiQA",  7.88,  7.48,  8.12,  8.00,  8.56,  9.18),
     ("HotPotQA",      21.35, 21.18, 22.11, 22.10, 23.23, 21.90),
     ("PoPQA",         16.23, 16.44, 15.48, 16.20, 16.38, 16.97)])


# tab:sft -- the same fourteen benchmarks after one round of supervised finetuning
SFT = _table(
    ("deepseek_mtp1", "echo_mtp1.5", "echo_mtp2_d7"),
    [("GSM8K",         61.87, 62.70, 65.20),
     ("MATH",          31.05, 38.54, 38.27),
     ("BBH",           54.41, 54.24, 58.06),
     ("BBEH",          11.30, 10.33, 10.00),
     ("PIQA",          74.49, 74.38, 74.38),
     ("WinoGrande",    61.17, 61.09, 61.56),
     ("MMLU",          49.17, 52.46, 55.53),
     ("MMLU-Pro",      29.06, 32.25, 31.28),
     ("HumanEval+",    47.56, 48.78, 51.83),
     ("MBPP+",         44.97, 51.59, 46.30),
     ("LiveCodeBench",  8.75, 11.25,  9.75),
     ("PreciseWikiQA",  4.22,  4.90,  5.06),
     ("HotPotQA",      19.04, 16.82, 19.16),
     ("PoPQA",         10.69, 10.94, 10.77)])
