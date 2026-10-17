#!/usr/bin/env bash
set -euo pipefail

# Launch one instance on each node. Example:
#   MPI_HOSTS=<HOST_IP>,<HOST_IP>,...,<HOST_IP>
#   MASTER_ADDR=<HOST_IP>
#   export MASTER_ADDR NNODES=12
#   mpirun -n 12 --host "$MPI_HOSTS" -npernode 1 \
#     -x MASTER_ADDR -x NNODES bash -c \
#     bash ${WORKSPACE:?set WORKSPACE}/SpecForge/examples/run_collect_perfectblend_12node.sh

VENV_PATH=${VENV_PATH:-${WORKSPACE:?set WORKSPACE}/.venvs/h800-existing-cu13}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}

NNODES=${NNODES:-12}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-}}
: "${NODE_RANK:?NODE_RANK or OMPI_COMM_WORLD_RANK must be set on every node}"
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set to rank-0 node}
MASTER_PORT=${MASTER_PORT:-29672}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

MODEL_PATH=${MODEL_PATH:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WORKSPACE:?set WORKSPACE}/paddle_hf_alignment/tokenizer_hf}
INPUT_FILE_PATH=${INPUT_FILE_PATH:-${WORKSPACE:?set WORKSPACE}/echo_merge_sources/perffix_full/z_ds_echo/data/perfectblend_deepseek_v4_flash_chat_t0_8k.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE:?set WORKSPACE}/perfectblend_step5000_raw}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-3072}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
BATCH_SIZE=${BATCH_SIZE:-24}
MTP_DEPTH=${MTP_DEPTH:-7}
LOG_INTERVAL=${LOG_INTERVAL:-10}
PROFILE=${PROFILE:-0}

mkdir -p "$OUTPUT_DIR"
echo "[the target PerfectBlend collector] node=$NODE_RANK/$NNODES world=$((NNODES * NPROC_PER_NODE)) gpu_per_node=$NPROC_PER_NODE input=$INPUT_FILE_PATH output=$OUTPUT_DIR"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
PROFILE_ARGS=()
if [[ "$PROFILE" == "1" ]]; then
  PROFILE_ARGS+=(--profile)
fi

"$VENV_PATH/bin/python" -m torch.distributed.run \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$ROOT_DIR/scripts/collect_perfectblend_sglang.py" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --input-file-path "$INPUT_FILE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --batch-size "$BATCH_SIZE" \
  --mtp-depth "$MTP_DEPTH" \
  --log-interval "$LOG_INTERVAL" \
  "${PROFILE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
