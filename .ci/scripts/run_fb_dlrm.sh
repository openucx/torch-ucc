#!/bin/bash -eEx
set -o pipefail

# TODO debug
exit 0

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

case ${DLRM_MODEL} in
"big")
    emb_size="1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000-1000"
    emb_dim="256"
    emb_lookup="100"
    bot_mlp="512-512-256"
    top_mlp="1024-1024-1024-1"
    loss_func="mse"
    round_targets="False"
    lr="0.01"
    #mb_size="2048"
    emb_lookup_fixed="0"
    ;;
"small")
    emb_size="1000-1000-1000-1000-1000-1000-1000-1000"
    emb_dim="64"
    emb_lookup="100"
    bot_mlp="512-512-64"
    top_mlp="1024-1024-1024-1"
    loss_func="mse"
    round_targets="False"
    lr="0.01"
    #mb_size="2048"
    emb_lookup_fixed="0"
    ;;
*)
    echo "ERROR: unsupported or empty DLRM_MODEL (${DLRM_MODEL})"
    exit 1
    ;;
esac

cd "${TORCH_UCC_ROOT_DIR}/workloads/dlrm"

MPIRUN_OPTIONS="\
    -np $NP \
    -H $HOSTS \
    --map-by node \
    -x LD_LIBRARY_PATH \
    --allow-run-as-root \
"

# shellcheck disable=SC2086
mpirun ${MPIRUN_OPTIONS} hostname
#mpirun ${MPIRUN_OPTIONS} python dlrm_s_pytorch.py \
#    --mini-batch-size=2048 \
#    --test-mini-batch-size=16384 \
#    --test-num-workers=0 \
#    --num-batches=100 \
#    --data-generation=random \
#    --arch-mlp-bot=$bot_mlp \
#    --arch-mlp-top=$top_mlp \
#    --arch-sparse-feature-size=$emb_dim \
#    --arch-embedding-size=$emb_size \
#    --num-indices-per-lookup=$emb_lookup \
#    --num-indices-per-lookup-fixed=$emb_lookup_fixed \
#    --arch-interaction-op=dot \
#    --numpy-rand-seed=727 \
#    --print-freq=1 \
#    --loss-function=$loss_func \
#    --round-targets=$round_targets \
#    --learning-rate=$lr \
#    --print-time \
#    --dist-backend=ucc \
#    --use-gpu
