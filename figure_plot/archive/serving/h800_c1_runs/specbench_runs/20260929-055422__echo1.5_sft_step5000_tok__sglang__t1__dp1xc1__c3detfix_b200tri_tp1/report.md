# specbench comparison report

> tok/s is only comparable **within** a harness. `accept_length` is comparable across harnesses.

## Harness: `sglang`

| method | draft ckpt | dataset | conc/replica | global conc | accept_len | tok/s | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | aime25 | 1 | 1 | N/A | 261.20 | N/A |
| baseline | - | alpaca | 1 | 1 | N/A | 260.08 | N/A |
| baseline | - | arena-hard | 1 | 1 | N/A | 262.03 | N/A |
| baseline | - | gsm8k | 1 | 1 | N/A | 257.33 | N/A |
| baseline | - | humaneval | 1 | 1 | N/A | 260.11 | N/A |
| baseline | - | livecodebench | 1 | 1 | N/A | 253.11 | N/A |
| baseline | - | math500 | 1 | 1 | N/A | 261.93 | N/A |
| baseline | - | mbpp | 1 | 1 | N/A | 264.42 | N/A |
| baseline | - | mt-bench | 1 | 1 | N/A | 261.21 | N/A |
| baseline | - | qa | 1 | 1 | N/A | 244.91 | N/A |
| baseline | - | rag | 1 | 1 | N/A | 228.19 | N/A |
| baseline | - | summarization | 1 | 1 | N/A | 241.23 | N/A |
| baseline | - | swe-bench | 1 | 1 | N/A | 258.43 | N/A |
| baseline | - | translation | 1 | 1 | N/A | 227.35 | N/A |
| dflash_b8 | serve_dflash_b8_s88785 | aime25 | 1 | 1 | 2.656 | 380.26 | 1.456 |
| dflash_b8 | serve_dflash_b8_s88785 | alpaca | 1 | 1 | 1.985 | 319.47 | 1.228 |
| dflash_b8 | serve_dflash_b8_s88785 | arena-hard | 1 | 1 | 2.308 | 351.81 | 1.343 |
| dflash_b8 | serve_dflash_b8_s88785 | gsm8k | 1 | 1 | 3.189 | 479.61 | 1.864 |
| dflash_b8 | serve_dflash_b8_s88785 | humaneval | 1 | 1 | 3.524 | 520.80 | 2.002 |
| dflash_b8 | serve_dflash_b8_s88785 | livecodebench | 1 | 1 | 2.799 | 340.80 | 1.346 |
| dflash_b8 | serve_dflash_b8_s88785 | math500 | 1 | 1 | 3.241 | 490.95 | 1.874 |
| dflash_b8 | serve_dflash_b8_s88785 | mbpp | 1 | 1 | 3.792 | 593.93 | 2.246 |
| dflash_b8 | serve_dflash_b8_s88785 | mt-bench | 1 | 1 | 2.347 | 369.58 | 1.415 |
| dflash_b8 | serve_dflash_b8_s88785 | qa | 1 | 1 | 1.759 | 268.30 | 1.095 |
| dflash_b8 | serve_dflash_b8_s88785 | rag | 1 | 1 | 2.510 | 249.72 | 1.094 |
| dflash_b8 | serve_dflash_b8_s88785 | summarization | 1 | 1 | 1.844 | 207.71 | 0.861 |
| dflash_b8 | serve_dflash_b8_s88785 | swe-bench | 1 | 1 | 2.148 | 284.25 | 1.100 |
| dflash_b8 | serve_dflash_b8_s88785 | translation | 1 | 1 | 1.860 | 269.46 | 1.185 |
| dspark_b7 | serve_dspark_b7_s88785 | aime25 | 1 | 1 | 2.920 | 389.55 | 1.491 |
| dspark_b7 | serve_dspark_b7_s88785 | alpaca | 1 | 1 | 2.150 | 319.29 | 1.228 |
| dspark_b7 | serve_dspark_b7_s88785 | arena-hard | 1 | 1 | 2.425 | 321.19 | 1.226 |
| dspark_b7 | serve_dspark_b7_s88785 | gsm8k | 1 | 1 | 3.302 | 463.09 | 1.800 |
| dspark_b7 | serve_dspark_b7_s88785 | humaneval | 1 | 1 | 4.143 | 561.02 | 2.157 |
| dspark_b7 | serve_dspark_b7_s88785 | livecodebench | 1 | 1 | 3.267 | 375.70 | 1.484 |
| dspark_b7 | serve_dspark_b7_s88785 | math500 | 1 | 1 | 3.422 | 477.78 | 1.824 |
| dspark_b7 | serve_dspark_b7_s88785 | mbpp | 1 | 1 | 3.979 | 579.61 | 2.192 |
| dspark_b7 | serve_dspark_b7_s88785 | mt-bench | 1 | 1 | 2.424 | 354.57 | 1.357 |
| dspark_b7 | serve_dspark_b7_s88785 | qa | 1 | 1 | 1.708 | 247.45 | 1.010 |
| dspark_b7 | serve_dspark_b7_s88785 | rag | 1 | 1 | 2.612 | 249.94 | 1.095 |
| dspark_b7 | serve_dspark_b7_s88785 | summarization | 1 | 1 | 1.916 | 201.53 | 0.835 |
| dspark_b7 | serve_dspark_b7_s88785 | swe-bench | 1 | 1 | 2.178 | 270.52 | 1.047 |
| dspark_b7 | serve_dspark_b7_s88785 | translation | 1 | 1 | 2.335 | 276.04 | 1.214 |

