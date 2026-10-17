#!/usr/bin/env bash
set -euo pipefail

# Launch one instance of this script per node. Example:
#   HOSTFILE=${WORKSPACE:?set WORKSPACE}/hostfile
#   MASTER_ADDR=$(awk 'NR==1{print $1}' "$HOSTFILE")
#   export MASTER_ADDR NNODES=6
#   mpirun -n 6 --hostfile "$HOSTFILE" -npernode 1 \
#     -x MASTER_ADDR -x NNODES bash -c \
#     'NODE_RANK=$OMPI_COMM_WORLD_RANK bash examples/run_target_sft_step5000_dspark_sglang_hf.sh'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

MACHINE_NAME=$(hostname -s)
VENV_PATH=${VENV_PATH:-${WORKSPACE:?set WORKSPACE}/.venvs/h800-existing-cu13}
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="${WORKSPACE:?set WORKSPACE}/SpecStack/third_party/sglang/python:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

NNODES=${NNODES:-6}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-}}
: "${NODE_RANK:?NODE_RANK or OMPI_COMM_WORLD_RANK must be set on every node}"
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set to rank-0 node}
MASTER_PORT=${MASTER_PORT:-29652}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WORKSPACE:?set WORKSPACE}/paddle_hf_alignment/tokenizer_hf}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-${WORKSPACE:?set WORKSPACE}/perfectblend_step5000_h5/perfectblend_step5000.zy}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000_dspark_6node}
CACHE_DIR=${CACHE_DIR:-${WORKSPACE:?set WORKSPACE}/specforge_cache_target_sft_step5000_dspark}

BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
# Draft budget. DSpark labels are anchor+1..anchor+block_size (specforge/core/dspark.py:149),
# so block_size IS the draft token count -- there is no -1 like train_dflash.py's
# non-anchor_prediction path. The target then verifies block_size+1 positions (the
# anchor/bonus token plus every draft token). BLOCK_SIZE=7 is the apples-to-apples
# setting against eagle3's ttt_length=7 (7 draft tokens + 1 bonus = 8 verified);
# 16 hands the draft 2.29x the budget.
BLOCK_SIZE=${BLOCK_SIZE:-16}
TP_SIZE=${TP_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-3}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
echo "[the target DSpark/sglang] node=$NODE_RANK/$NNODES world=$((NNODES * NPROC_PER_NODE)) tp=$TP_SIZE dp=$((NNODES * NPROC_PER_NODE / TP_SIZE)) local_batch=$BATCH_SIZE global_batch=$((NNODES * NPROC_PER_NODE * BATCH_SIZE * ACCUMULATION_STEPS)) block_size=$BLOCK_SIZE draft_tokens=$BLOCK_SIZE verified=$((BLOCK_SIZE + 1))"

"$VENV_PATH/bin/python" -m torch.distributed.run \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$ROOT_DIR/scripts/train_dspark.py" \
  --target-model-path "$TARGET_MODEL_PATH" \
  --target-model-backend sglang \
  --sglang-mem-fraction-static "${MEM_FRACTION:-0.10}" \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-fa3}" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate 6e-4 \
  --warmup-ratio 0.04 \
  --max-grad-norm 1.0 \
  --max-length "$MAX_LENGTH" \
  --block-size "$BLOCK_SIZE" \
  --num-draft-layers 5 \
  --num-anchors 512 \
  --loss-decay-gamma 4.0 \
  --ce-loss-alpha 0.1 \
  --l1-loss-alpha 0.9 \
  --confidence-head-alpha 1.0 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --mask-token-id 2 \
  --attention-backend flex_attention \
  --dataloader-num-workers 2 \
  --log-interval 20 \
  --save-schedule "0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0" \
  --resume \
  --report-to none \
  --tp-size "$TP_SIZE" \
  --dist-timeout 120
