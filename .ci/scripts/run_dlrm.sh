#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

TORCH_UCC_MODE="$1"
CPU_GPU_MODE="$2"
HOSTFILE="$3"

if [ "${TORCH_UCC_MODE}" != "ucc" ] && [ "${TORCH_UCC_MODE}" != "xccl" ]; then
    echo "ERROR: unsupported or empty TORCH_UCC_MODE (${TORCH_UCC_MODE}), supported values: ucc, xccl"
    exit 1
fi

export TORCH_UCC_MODE
export CPU_GPU_MODE

if [ -z "$HOSTFILE" ]; then
    echo "ERROR: HOSTFILE is not specified"
    exit 1
fi

export PATH="/usr/lib64/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH}"

HEAD_NODE=$(head -1 "$HOSTFILE")
export HEAD_NODE
export MASTER_ADDR=${HEAD_NODE}

NP=$(wc --lines "$HOSTFILE" | awk '{print $1}')

# shellcheck disable=SC2086
mpirun \
    -np $NP \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    hostname

# shellcheck disable=SC2086
mpirun \
    -np $NP \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    cat /proc/1/cgroup

# shellcheck disable=SC2086
mpirun \
    -np $NP \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x MASTER_ADDR \
    -x TORCH_UCC_MODE \
    -x CPU_GPU_MODE \
    /opt/nvidia/torch-ucc/src/.ci/scripts/run_dlrm_s_pytorch.sh
