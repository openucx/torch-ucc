#!/bin/bash -eEx
set -o pipefail

command -v mpirun
export TORCH_UCC_XCCL_TLS=ucx
export UCX_WARN_UNUSED_ENV_VARS=n
ucx_info -e -u t
TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT="${TORCH_UCC_SRC_DIR}_xccl"

echo "UCC barrier"
/bin/bash ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/start_test.sh ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/torch_barrier_test.py --backend=gloo

echo "UCC alltoall"
/bin/bash ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/start_test.sh ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/torch_alltoall_test.py --backend=gloo

echo "UCC alltoallv"
/bin/bash ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/start_test.sh ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/torch_alltoallv_test.py --backend=gloo

echo "UCC allgather"
/bin/bash ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/start_test.sh ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/torch_allgather_test.py --backend=gloo

echo "UCC allreduce"
/bin/bash ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/start_test.sh ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/torch_allreduce_test.py --backend=gloo

echo "UCC broadcast"
/bin/bash ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/start_test.sh ${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/test/torch_bcast_test.py --backend=gloo
