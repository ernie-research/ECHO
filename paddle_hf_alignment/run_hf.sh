#!/usr/bin/env bash
set -euo pipefail

DELIVERY=$(dirname "$(readlink -f "$0")")
ECHO=${ECHO_CKPT:-<WORKSPACE>/checkpoints/echo_mtp_step_78000/extracted/hf_checkpoint-78000}
BASELINE=${BASELINE_CKPT:-<WORKSPACE>/checkpoints/moe_mtp_baseline_step_78000/extracted/hf_checkpoint-78000}
DATA=$DELIVERY/data/gsm8k_test_completion_first50.jsonl

MODEL=${1:-echo}
GPU=${2:-0}
TEMP=${3:-1.0}
# temp=0 is invalid here (softmax divides by temp -> NaN); 1e-6 == argmax/greedy.
if [[ "$TEMP" == "0" || "$TEMP" == "0.0" ]]; then
  echo "[warn] temp=0 invalid (softmax /temp -> NaN); using 1e-6 (greedy/argmax)." >&2
  TEMP=1e-6
fi
if [[ "$MODEL" == "echo" ]]; then
  CKPT=$ECHO
elif [[ "$MODEL" == "baseline" ]]; then
  CKPT=$BASELINE
else
  echo "Usage: $0 {echo|baseline} [physical_gpu_id] [temp=1.0|0]" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES=$GPU python "$DELIVERY/eval/eval_stochastic_hf.py" \
  --ckpt "$CKPT" \
  --dataset "$DATA" \
  --out "$DELIVERY/results/stochastic_hf_${MODEL}_50_t${TEMP}_rerun.json" \
  --gpu 0 --n 50 --max-new 64 --K 3 --temp "$TEMP"
