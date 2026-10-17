# specbench comparison report

> tok/s is only comparable **within** a harness. `accept_length` is comparable across harnesses.

## Harness: `sglang`

| method | draft ckpt | dataset | conc/replica | global conc | accept_len | tok/s | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | alpaca | 16 | 16 | N/A | 151.07 | N/A |
| baseline | - | alpaca | 32 | 32 | N/A | 107.74 | N/A |
| baseline | - | alpaca | 64 | 64 | N/A | 89.21 | N/A |
| baseline | - | arena-hard | 16 | 16 | N/A | 198.73 | N/A |
| baseline | - | arena-hard | 32 | 32 | N/A | 161.68 | N/A |
| baseline | - | arena-hard | 64 | 64 | N/A | 148.14 | N/A |
| baseline | - | gsm8k | 16 | 16 | N/A | 161.47 | N/A |
| baseline | - | gsm8k | 32 | 32 | N/A | 112.21 | N/A |
| baseline | - | gsm8k | 64 | 64 | N/A | 81.21 | N/A |
| baseline | - | math500 | 16 | 16 | N/A | 190.25 | N/A |
| baseline | - | math500 | 32 | 32 | N/A | 147.51 | N/A |
| baseline | - | math500 | 64 | 64 | N/A | 118.64 | N/A |
| dspark_b16 | serve_dspark_b16_s88785 | alpaca | 16 | 16 | 3.979 | 152.99 | 1.013 |
| dspark_b16 | serve_dspark_b16_s88785 | alpaca | 32 | 32 | 3.956 | 96.76 | 0.898 |
| dspark_b16 | serve_dspark_b16_s88785 | alpaca | 64 | 64 | 4.019 | 65.09 | 0.730 |
| dspark_b16 | serve_dspark_b16_s88785 | arena-hard | 16 | 16 | 4.392 | 201.41 | 1.013 |
| dspark_b16 | serve_dspark_b16_s88785 | arena-hard | 32 | 32 | 4.415 | 126.47 | 0.782 |
| dspark_b16 | serve_dspark_b16_s88785 | arena-hard | 64 | 64 | 4.443 | 80.66 | 0.544 |
| dspark_b16 | serve_dspark_b16_s88785 | gsm8k | 16 | 16 | 8.024 | 230.17 | 1.425 |
| dspark_b16 | serve_dspark_b16_s88785 | gsm8k | 32 | 32 | 7.932 | 157.06 | 1.400 |
| dspark_b16 | serve_dspark_b16_s88785 | gsm8k | 64 | 64 | 7.895 | 109.69 | 1.351 |
| dspark_b16 | serve_dspark_b16_s88785 | math500 | 16 | 16 | 7.439 | 279.87 | 1.471 |
| dspark_b16 | serve_dspark_b16_s88785 | math500 | 32 | 32 | 7.381 | 174.31 | 1.182 |
| dspark_b16 | serve_dspark_b16_s88785 | math500 | 64 | 64 | 7.360 | 113.10 | 0.953 |
| dspark_b7 | serve_dspark_b7_s88785 | alpaca | 16 | 16 | 3.795 | 189.33 | 1.253 |
| dspark_b7 | serve_dspark_b7_s88785 | alpaca | 32 | 32 | 3.759 | 119.55 | 1.110 |
| dspark_b7 | serve_dspark_b7_s88785 | alpaca | 64 | 64 | 3.766 | 85.13 | 0.954 |
| dspark_b7 | serve_dspark_b7_s88785 | arena-hard | 16 | 16 | 4.047 | 261.39 | 1.315 |
| dspark_b7 | serve_dspark_b7_s88785 | arena-hard | 32 | 32 | 4.050 | 170.48 | 1.054 |
| dspark_b7 | serve_dspark_b7_s88785 | arena-hard | 64 | 64 | 4.053 | 111.61 | 0.753 |
| dspark_b7 | serve_dspark_b7_s88785 | gsm8k | 16 | 16 | 6.057 | 237.90 | 1.473 |
| dspark_b7 | serve_dspark_b7_s88785 | gsm8k | 32 | 32 | 6.040 | 166.04 | 1.480 |
| dspark_b7 | serve_dspark_b7_s88785 | gsm8k | 64 | 64 | 6.003 | 117.52 | 1.447 |
| dspark_b7 | serve_dspark_b7_s88785 | math500 | 16 | 16 | 5.780 | 303.74 | 1.597 |
| dspark_b7 | serve_dspark_b7_s88785 | math500 | 32 | 32 | 5.873 | 197.69 | 1.340 |
| dspark_b7 | serve_dspark_b7_s88785 | math500 | 64 | 64 | 5.858 | 136.06 | 1.147 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | alpaca | 16 | 16 | 3.729 | 194.56 | 1.288 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | alpaca | 32 | 32 | 3.666 | 133.17 | 1.236 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | alpaca | 64 | 64 | 3.733 | 105.05 | 1.178 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | arena-hard | 16 | 16 | 3.824 | 282.31 | 1.421 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | arena-hard | 32 | 32 | 3.834 | 198.56 | 1.228 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | arena-hard | 64 | 64 | 3.818 | 154.79 | 1.045 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | gsm8k | 16 | 16 | 5.710 | 248.07 | 1.536 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | gsm8k | 32 | 32 | 5.706 | 171.28 | 1.526 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | gsm8k | 64 | 64 | 5.710 | 138.54 | 1.706 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | math500 | 16 | 16 | 5.725 | 327.78 | 1.723 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | math500 | 32 | 32 | 5.725 | 231.89 | 1.572 |
| eagle3_cold_final | serve_eagle3_cold_s88785_v2 | math500 | 64 | 64 | 5.718 | 175.00 | 1.475 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | alpaca | 16 | 16 | 3.692 | 193.51 | 1.281 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | alpaca | 32 | 32 | 3.627 | 127.13 | 1.180 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | alpaca | 64 | 64 | 3.665 | 94.97 | 1.065 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | arena-hard | 16 | 16 | 4.061 | 288.11 | 1.450 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | arena-hard | 32 | 32 | 4.030 | 201.70 | 1.248 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | arena-hard | 64 | 64 | 4.017 | 148.92 | 1.005 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | gsm8k | 16 | 16 | 4.993 | 240.41 | 1.489 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | gsm8k | 32 | 32 | 4.969 | 154.12 | 1.373 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | gsm8k | 64 | 64 | 5.001 | 134.68 | 1.658 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | math500 | 16 | 16 | 5.156 | 312.15 | 1.641 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | math500 | 32 | 32 | 5.150 | 210.15 | 1.425 |
| echo_mtp | echo1.5_sft_step5000_nextn_draft | math500 | 64 | 64 | 5.161 | 146.97 | 1.239 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 16 | 16 | 2.892 | 197.10 | 1.305 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 32 | 32 | 2.906 | 129.66 | 1.203 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | alpaca | 64 | 64 | 2.909 | 95.65 | 1.072 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 16 | 16 | 3.071 | 290.97 | 1.464 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 32 | 32 | 3.081 | 207.65 | 1.284 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | arena-hard | 64 | 64 | 3.062 | 159.34 | 1.076 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 16 | 16 | 3.445 | 218.93 | 1.356 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 32 | 32 | 3.446 | 156.60 | 1.396 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | gsm8k | 64 | 64 | 3.445 | 128.63 | 1.584 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 16 | 16 | 3.484 | 296.97 | 1.561 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 32 | 32 | 3.477 | 203.38 | 1.379 |
| echo_mtp_k3 | echo1.5_sft_step5000_nextn_draft | math500 | 64 | 64 | 3.472 | 148.31 | 1.250 |

