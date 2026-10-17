# specbench comparison report

> tok/s is only comparable **within** a harness. `accept_length` is comparable across harnesses.

## Harness: `sglang`

| method | draft ckpt | dataset | conc/replica | global conc | accept_len | tok/s | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | aime25 | 1 | 1 | 2.903 | 645.46 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | aime25 | 4 | 4 | 2.921 | 501.45 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | aime25 | 8 | 8 | 2.903 | 406.93 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | aime25 | 16 | 16 | 2.878 | 351.51 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 1 | 1 | 2.392 | 499.20 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 4 | 4 | 2.366 | 337.00 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 8 | 8 | 2.360 | 265.17 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 16 | 16 | 2.335 | 190.62 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 1 | 1 | 2.614 | 580.40 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 4 | 4 | 2.547 | 431.68 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 8 | 8 | 2.551 | 350.36 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 16 | 16 | 2.558 | 273.64 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 1 | 1 | 3.032 | 624.46 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 4 | 4 | 3.012 | 410.43 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 8 | 8 | 3.010 | 327.77 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 16 | 16 | 2.977 | 232.07 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | humaneval | 1 | 1 | 3.161 | 677.58 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | humaneval | 4 | 4 | 3.172 | 494.79 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | humaneval | 8 | 8 | 3.175 | 379.90 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | humaneval | 16 | 16 | 3.195 | 280.38 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | livecodebench | 1 | 1 | 2.990 | 604.41 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | livecodebench | 4 | 4 | 3.068 | 420.76 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | livecodebench | 8 | 8 | 3.024 | 308.74 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | livecodebench | 16 | 16 | 3.069 | 228.24 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 1 | 1 | 3.093 | 668.29 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 4 | 4 | 3.078 | 487.72 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 8 | 8 | 3.073 | 375.59 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 16 | 16 | 3.087 | 285.06 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mbpp | 1 | 1 | 3.137 | 687.77 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mbpp | 4 | 4 | 3.127 | 513.89 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mbpp | 8 | 8 | 3.135 | 411.47 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mbpp | 16 | 16 | 3.159 | 322.85 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mt-bench | 1 | 1 | 2.450 | 524.04 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mt-bench | 4 | 4 | 2.421 | 373.07 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mt-bench | 8 | 8 | 2.449 | 286.15 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | mt-bench | 16 | 16 | 2.463 | 215.56 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | qa | 1 | 1 | 2.161 | 390.68 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | qa | 4 | 4 | 2.276 | 230.35 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | qa | 8 | 8 | 2.167 | 156.87 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | qa | 16 | 16 | 2.146 | 114.44 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | rag | 1 | 1 | 2.919 | 397.63 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | rag | 4 | 4 | 2.862 | 196.39 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | rag | 8 | 8 | 2.876 | 136.33 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | rag | 16 | 16 | 2.858 | 103.35 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | summarization | 1 | 1 | 2.111 | 402.44 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | summarization | 4 | 4 | 2.148 | 266.96 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | summarization | 8 | 8 | 2.138 | 191.76 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | summarization | 16 | 16 | 2.163 | 138.12 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | swe-bench | 1 | 1 | 2.871 | 627.38 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | swe-bench | 4 | 4 | 2.861 | 460.27 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | swe-bench | 8 | 8 | 2.890 | 383.87 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | swe-bench | 16 | 16 | 2.940 | 300.24 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | translation | 1 | 1 | 2.466 | 311.58 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | translation | 4 | 4 | 2.384 | 134.43 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | translation | 8 | 8 | 2.409 | 91.74 | N/A |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | translation | 16 | 16 | 2.399 | 68.72 | N/A |

