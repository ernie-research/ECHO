#!/usr/bin/env bash
# 12 nodes split 6 / 3 / 3, launched by one mpirun:
#   mpirun -n 12 -npernode 1 bash ${WORKSPACE:?set WORKSPACE}/launch_split_6_3_3.sh
#
#   ranks 0-5   dspark        6 nodes, tp=8 -> dp=6, B=1 -> global batch 48
#   ranks 6-8   eagle3 cold   3 nodes, tp=8 -> dp=3, B=2 -> global batch 48
#   ranks 9-11  eagle3 warm   3 nodes, tp=8 -> dp=3, B=2 -> global batch 48
#
# All three land on the SAME global batch 48 (= 29,595 steps/epoch), which makes
# their loss/acc curves directly comparable per step.
#
# BATCH_SIZE stays per-group: dspark's draft is 2.22B (5 the target decoder layers +
# markov + confidence heads) and measures 55 GiB/card at B=1, so B=2 OOMs it
# (>79 GiB). eagle3's draft is 0.67B (one MTP block): 26 GiB at B=1, 47 GiB at
# B=2, ~68 GiB at B=3 (too close to the flex-attention recompile headroom that
# killed a node earlier), ~88 GiB at B=4 (impossible).
#
# All groups: the target-typed draft (per-family layer type) + sglang TP8 target with
# the 2026-09-18 fixes (RoPE even/odd regroup, fp32 router GEMM, fp32
# e_score_correction_bias, --shard-target-output, mem_fraction 0.10).
GROUP_A_METHOD=${GROUP_A_METHOD:-dspark}       # ranks 0-5
GROUP_B_METHOD=${GROUP_B_METHOD:-eagle3}       # ranks 6-8
GROUP_C_METHOD=${GROUP_C_METHOD:-eagle3_warm}  # ranks 9-11

R=${OMPI_COMM_WORLD_RANK:-0}
export NPROC_PER_NODE=8
export DATALOADER_NUM_WORKERS=4
export MEM_FRACTION=0.10
export TP_SIZE=8

if [ "$R" -le 5 ]; then
  METHOD=$GROUP_A_METHOD
  export NNODES=6 NODE_RANK=$R BATCH_SIZE=1
  export MASTER_ADDR=<HOST_IP> MASTER_PORT=29872
elif [ "$R" -le 8 ]; then
  METHOD=$GROUP_B_METHOD
  export NNODES=3 NODE_RANK=$((R - 6)) BATCH_SIZE=2
  export MASTER_ADDR=<HOST_IP> MASTER_PORT=29880
else
  METHOD=$GROUP_C_METHOD
  # rank 9 (<HOST_IP>) is the rendezvous; rank 10 is 245.80 = node 2087, the
  # card that repeatedly lost its CUDA context, so it must not be the master.
  export NNODES=3 NODE_RANK=$((R - 9)) BATCH_SIZE=2
  export MASTER_ADDR=<HOST_IP> MASTER_PORT=29890
fi

if [ "$METHOD" = "skip" ]; then
  echo "rank=$R skipped"
  exit 0
fi

LOGDIR=${WORKSPACE:?set WORKSPACE}/logs_3way/$METHOD
mkdir -p "$LOGDIR"
cd ${WORKSPACE:?set WORKSPACE}   # .zy manifest holds relative h5 paths

case $METHOD in
  eagle3)
    export OUTPUT_DIR=${WORKSPACE:?set WORKSPACE}/run4n_eagle3_cold
    SCRIPT=SpecForge/examples/run_target_sft_step5000_eagle3_sglang.sh ;;
  dspark)
    export OUTPUT_DIR=${WORKSPACE:?set WORKSPACE}/run4n_dspark
    SCRIPT=SpecForge/examples/run_target_sft_step5000_dspark_sglang.sh ;;
  eagle3_warm)
    export OUTPUT_DIR=${WORKSPACE:?set WORKSPACE}/run4n_eagle3_warm
    SCRIPT=SpecForge/examples/run_target_sft_step5000_eagle3_warm_sglang.sh ;;
esac

setsid nohup bash "${WORKSPACE:?set WORKSPACE}/$SCRIPT" \
  >>"$LOGDIR/node${NODE_RANK}.log" 2>&1 </dev/null &
echo "rank=$R method=$METHOD nnodes=$NNODES node_rank=$NODE_RANK bs=$BATCH_SIZE"
exit 0
