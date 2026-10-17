# specbench comparison report

> tok/s is only comparable **within** a harness. `accept_length` is comparable across harnesses.

## Harness: `sglang`

| method | draft ckpt | dataset | conc/replica | global conc | accept_len | tok/s | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | aime25 | 1 | 1 | N/A | 369.64 | N/A |
| baseline | - | aime25 | 4 | 4 | N/A | 308.88 | N/A |
| baseline | - | aime25 | 8 | 8 | N/A | 289.03 | N/A |
| baseline | - | aime25 | 16 | 16 | N/A | 267.49 | N/A |
| baseline | - | alpaca | 1 | 1 | N/A | 356.78 | N/A |
| baseline | - | alpaca | 4 | 4 | N/A | 266.74 | N/A |
| baseline | - | alpaca | 8 | 8 | N/A | 215.06 | N/A |
| baseline | - | alpaca | 16 | 16 | N/A | 181.27 | N/A |
| baseline | - | arena-hard | 1 | 1 | N/A | 369.88 | N/A |
| baseline | - | arena-hard | 4 | 4 | N/A | 303.52 | N/A |
| baseline | - | arena-hard | 8 | 8 | N/A | 266.44 | N/A |
| baseline | - | arena-hard | 16 | 16 | N/A | 230.10 | N/A |
| baseline | - | gsm8k | 1 | 1 | N/A | 357.51 | N/A |
| baseline | - | gsm8k | 4 | 4 | N/A | 277.49 | N/A |
| baseline | - | gsm8k | 8 | 8 | N/A | 232.15 | N/A |
| baseline | - | gsm8k | 16 | 16 | N/A | 182.14 | N/A |
| baseline | - | humaneval | 1 | 1 | N/A | 363.49 | N/A |
| baseline | - | humaneval | 4 | 4 | N/A | 295.79 | N/A |
| baseline | - | humaneval | 8 | 8 | N/A | 246.95 | N/A |
| baseline | - | humaneval | 16 | 16 | N/A | 197.07 | N/A |
| baseline | - | livecodebench | 1 | 1 | N/A | 352.19 | N/A |
| baseline | - | livecodebench | 4 | 4 | N/A | 272.17 | N/A |
| baseline | - | livecodebench | 8 | 8 | N/A | 221.40 | N/A |
| baseline | - | livecodebench | 16 | 16 | N/A | 175.88 | N/A |
| baseline | - | math500 | 1 | 1 | N/A | 366.57 | N/A |
| baseline | - | math500 | 4 | 4 | N/A | 297.60 | N/A |
| baseline | - | math500 | 8 | 8 | N/A | 251.77 | N/A |
| baseline | - | math500 | 16 | 16 | N/A | 209.25 | N/A |
| baseline | - | mbpp | 1 | 1 | N/A | 371.06 | N/A |
| baseline | - | mbpp | 4 | 4 | N/A | 304.97 | N/A |
| baseline | - | mbpp | 8 | 8 | N/A | 273.08 | N/A |
| baseline | - | mbpp | 16 | 16 | N/A | 238.80 | N/A |
| baseline | - | mt-bench | 1 | 1 | N/A | 362.18 | N/A |
| baseline | - | mt-bench | 4 | 4 | N/A | 281.76 | N/A |
| baseline | - | mt-bench | 8 | 8 | N/A | 235.31 | N/A |
| baseline | - | mt-bench | 16 | 16 | N/A | 182.46 | N/A |
| baseline | - | qa | 1 | 1 | N/A | 312.85 | N/A |
| baseline | - | qa | 4 | 4 | N/A | 184.11 | N/A |
| baseline | - | qa | 8 | 8 | N/A | 143.08 | N/A |
| baseline | - | qa | 16 | 16 | N/A | 113.43 | N/A |
| baseline | - | rag | 1 | 1 | N/A | 284.94 | N/A |
| baseline | - | rag | 4 | 4 | N/A | 162.24 | N/A |
| baseline | - | rag | 8 | 8 | N/A | 114.27 | N/A |
| baseline | - | rag | 16 | 16 | N/A | 88.56 | N/A |
| baseline | - | summarization | 1 | 1 | N/A | 331.53 | N/A |
| baseline | - | summarization | 4 | 4 | N/A | 231.87 | N/A |
| baseline | - | summarization | 8 | 8 | N/A | 171.82 | N/A |
| baseline | - | summarization | 16 | 16 | N/A | 124.52 | N/A |
| baseline | - | swe-bench | 1 | 1 | N/A | 366.67 | N/A |
| baseline | - | swe-bench | 4 | 4 | N/A | 299.98 | N/A |
| baseline | - | swe-bench | 8 | 8 | N/A | 260.94 | N/A |
| baseline | - | swe-bench | 16 | 16 | N/A | 218.19 | N/A |
| baseline | - | translation | 1 | 1 | N/A | 258.38 | N/A |
| baseline | - | translation | 4 | 4 | N/A | 135.37 | N/A |
| baseline | - | translation | 8 | 8 | N/A | 92.96 | N/A |
| baseline | - | translation | 16 | 16 | N/A | 69.06 | N/A |

