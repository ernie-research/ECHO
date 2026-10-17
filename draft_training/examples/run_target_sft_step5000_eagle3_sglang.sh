#!/usr/bin/env bash
set -euo pipefail

# the target EAGLE-3 with the sglang TP8 target engine (instead of the replicated HF
# eager target). Same data/draft config as run_target_sft_step5000_eagle3_6node_hf.sh;
# only the target backend differs.
#
# Numerics for this path were reconciled on 2026-09-18: gate-on argmax agreement
# vs the HF engine is 99.61% on real text / 100% gate-off (ECHO/the port log (not published) §5).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

VENV_PATH=${VENV_PATH:-${WORKSPACE:?set WORKSPACE}/.venvs/h800-existing-cu13}
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# The ported the target sglang model lives in the SpecStack fork, which must win over
# any installed sglang.
export PYTHONPATH="${WORKSPACE:?set WORKSPACE}/SpecStack/third_party/sglang/python:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

NNODES=${NNODES:-12}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-}}
: "${NODE_RANK:?NODE_RANK or OMPI_COMM_WORLD_RANK must be set on every node}"
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set to rank-0 node}
MASTER_PORT=${MASTER_PORT:-29720}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WORKSPACE:?set WORKSPACE}/paddle_hf_alignment/tokenizer_hf}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-${WORKSPACE:?set WORKSPACE}/perfectblend_step5000_h5/perfectblend_step5000.zy}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000_eagle3_sglang}
CACHE_DIR=${CACHE_DIR:-${WORKSPACE:?set WORKSPACE}/specforge_cache_target_sft_step5000_eagle3}

BATCH_SIZE=${BATCH_SIZE:-1}
TP_SIZE=${TP_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-3}
# Weights are ~2.7GiB/rank at TP8; the rest of mem_fraction_static is the KV pool,
# which draft training barely uses (one extend forward, no decode cache).
MEM_FRACTION=${MEM_FRACTION:-0.10}
ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-fa3}

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
echo "[the target EAGLE3/sglang] node=$NODE_RANK/$NNODES world=$((NNODES * NPROC_PER_NODE)) tp=$TP_SIZE dp=$((NNODES * NPROC_PER_NODE / TP_SIZE)) local_batch=$BATCH_SIZE target_batch=$((TP_SIZE * BATCH_SIZE)) mem_fraction=$MEM_FRACTION attn=$ATTENTION_BACKEND"

"$VENV_PATH/bin/python" -m torch.distributed.run \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$ROOT_DIR/scripts/train_eagle3.py" \
  --target-model-path "$TARGET_MODEL_PATH" \
  --target-model-backend sglang \
  --sglang-mem-fraction-static "$MEM_FRACTION" \
  --sglang-attention-backend "$ATTENTION_BACKEND" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --draft-model-config "$ROOT_DIR/configs/target-eagle3.json" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate 1e-4 \
  --max-length "$MAX_LENGTH" \
  --ttt-length 7 \
  --attention-backend sdpa \
  --loss-type kl \
  --tp-size "$TP_SIZE" \
  --shard-target-output \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-2}" \
  --build-dataset-num-proc 1 \
  --dist-timeout 120 \
  --save-schedule "0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0,1.5,2.0,2.5,3.0" \
  --resume \
  --report-to none
