#!/bin/bash -eEx
set -o pipefail

command -v mpirun
export TORCH_UCC_XCCL_TLS=ucx
export UCX_WARN_UNUSED_ENV_VARS=n
ucx_info -e -u t
echo "XCCL allreduce"
/bin/bash ${TORCH_UCC_SRC_DIR}/test/start_test.sh ${TORCH_UCC_SRC_DIR}/test/torch_allreduce_test.py --backend=gloo
echo "XCCL alltoall"
/bin/bash ${TORCH_UCC_SRC_DIR}/test/start_test.sh ${TORCH_UCC_SRC_DIR}/test/torch_alltoall_test.py --backend=gloo
echo "XCCL alltoallv"
/bin/bash ${TORCH_UCC_SRC_DIR}/test/start_test.sh ${TORCH_UCC_SRC_DIR}/test/torch_alltoallv_test.py --backend=gloo
echo "XCCL barrier"
/bin/bash ${TORCH_UCC_SRC_DIR}/test/start_test.sh ${TORCH_UCC_SRC_DIR}/test/torch_barrier_test.py --backend=gloo
echo "XCCL allgather"
/bin/bash ${TORCH_UCC_SRC_DIR}/test/start_test.sh ${TORCH_UCC_SRC_DIR}/test/torch_allgather_test.py --backend=gloo
echo "XCCL broadcast"
/bin/bash ${TORCH_UCC_SRC_DIR}/test/start_test.sh ${TORCH_UCC_SRC_DIR}/test/torch_bcast_test.py --backend=gloo
