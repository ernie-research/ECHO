#!/usr/bin/env bash
set -euo pipefail

# the target Echo x DFlash (online, in-process SGLang target).
#
# Sibling of run_target_sft_step5000_dspark_sglang.sh: same target, same .zy
# dataset, same 5-route capture [1, 6, 11, 16, 21] (build_target_layer_ids(24, 5)),
# same the target decoder backbone (TargetDSparkLayer is reused by
# TargetDFlashDraftModel).  The only difference is the method: no Markov /
# confidence head, plain DFlash loss.
#
# Draft budget: BLOCK_SIZE=8 without --anchor-prediction -> position 0 predicts
# the anchor and is dropped from the loss, so 7 draft tokens are emitted and the
# target verifies 8 positions (train_dflash.py:614-629).  That is exactly the
# DSpark block-7 arm's budget (7 draft / 8 verified) and eagle3's ttt_length=7,
# so the three arms are comparable.
#
# Launch one instance per node, e.g.:
#   mpirun -n 2 --allow-run-as-root -npernode 1 -x MASTER_ADDR -x NNODES \
#     bash -c 'NODE_RANK=$OMPI_COMM_WORLD_RANK bash examples/run_target_sft_step5000_dflash_sglang.sh'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# B200 (sm100) needs the per-host specstack venv: torch 2.13.0+cu130 with an
# sm100 sgl_kernel.  The shared h800-existing-cu13 venv ships an sgl_kernel built
# against the torch 2.9 ABI and dies with `undefined symbol:
# _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib` on these cards.
VENV_PATH=${VENV_PATH:-/root/.venvs/$(hostname -s)/specstack}
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$HOME/.local/bin:$PATH"
# sglang must be the GPFS SpecStack checkout: it is the only tree with
# models/modeling.py.  PYTHONPATH wins over the venv's editable installs.
SGLANG_PY=${SGLANG_PY:-${WORKSPACE:?set WORKSPACE}/SpecStack/third_party/sglang/python}
export PYTHONPATH="$ROOT_DIR:$SGLANG_PY${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
# SGLANG_ATTENTION_BACKEND is deliberately unset by default: fa3 asserts
# `SM>=80 and SM<=90`, so it cannot run on B200 (sm100).  Leaving the flag off
# lets sglang pick its own sm100 default, which is what the DSV4 runs on these
# same cards do.  Set it explicitly (e.g. fa3) when running on H cards.

NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-}}
: "${NODE_RANK:?NODE_RANK or OMPI_COMM_WORLD_RANK must be set on every node}"
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set to the rank-0 node}
MASTER_PORT=${MASTER_PORT:-29680}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WORKSPACE:?set WORKSPACE}/paddle_hf_alignment/tokenizer_hf}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-${WORKSPACE:?set WORKSPACE}/perfectblend_step5000_h5/perfectblend_step5000.zy}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKSPACE:?set WORKSPACE}/echo1.5_sft_step5000_dflash}
# Tokenization cache is dataset-only (block/route independent), so it is shared
# with the dspark arm instead of rebuilt.
CACHE_DIR=${CACHE_DIR:-${WORKSPACE:?set WORKSPACE}/specforge_cache_target_sft_step5000_dspark}

BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
BLOCK_SIZE=${BLOCK_SIZE:-8}          # 8 block positions -> 7 draft tokens, 8 verified
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-5}
TP_SIZE=${TP_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-3}

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
echo "[the target DFlash/sglang] node=$NODE_RANK/$NNODES world=$((NNODES * NPROC_PER_NODE)) tp=$TP_SIZE dp=$((NNODES * NPROC_PER_NODE / TP_SIZE)) local_batch=$BATCH_SIZE global_batch=$((NNODES * NPROC_PER_NODE * BATCH_SIZE * ACCUMULATION_STEPS)) block_size=$BLOCK_SIZE draft_layers=$NUM_DRAFT_LAYERS draft_tokens=$((BLOCK_SIZE - 1)) verified=$BLOCK_SIZE"

"$VENV_PATH/bin/python" -m torch.distributed.run \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$ROOT_DIR/scripts/train_dflash.py" \
  --target-model-path "$TARGET_MODEL_PATH" \
  --target-model-backend sglang \
  --sglang-mem-fraction-static "${MEM_FRACTION:-0.10}" \
  ${SGLANG_ATTENTION_BACKEND:+--sglang-attention-backend "$SGLANG_ATTENTION_BACKEND"} \
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
  --num-draft-layers "$NUM_DRAFT_LAYERS" \
  --loss-type dflash \
  --num-anchors 512 \
  --loss-decay-gamma 4.0 \
  --mask-token-id 2 \
  --attention-backend flex_attention \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-2}" \
  --log-interval 20 \
  --save-schedule "0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0" \
  --resume \
  --tp-size "$TP_SIZE" \
  --dist-timeout 120
