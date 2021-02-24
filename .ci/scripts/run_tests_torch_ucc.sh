#!/bin/sh -eEx

command -v mpirun
export TORCH_UCC_XCCL_TLS=ucx
ucx_info -e -u t
export UCX_LOG_LEVEL=info
#echo "XCCL allreduce"
#/bin/bash ${XCCL_SRC_DIR}/test/start_test.sh ${XCCL_SRC_DIR}/test/torch_allreduce_test.py --backend=gloo
#echo "XCCL alltoall"
#/bin/bash ${XCCL_SRC_DIR}/test/start_test.sh ${XCCL_SRC_DIR}/test/torch_alltoall_test.py --backend=gloo
#echo "XCCL alltoallv"
#/bin/bash ${XCCL_SRC_DIR}/test/start_test.sh ${XCCL_SRC_DIR}/test/torch_alltoallv_test.py --backend=gloo
#echo "XCCL barrier"
#/bin/bash ${XCCL_SRC_DIR}/test/start_test.sh ${XCCL_SRC_DIR}/test/torch_barrier_test.py --backend=gloo
#echo "XCCL allgather"
#/bin/bash ${XCCL_SRC_DIR}/test/start_test.sh ${XCCL_SRC_DIR}/test/torch_allgather_test.py --backend=gloo
#echo "XCCL broadcast"
#/bin/bash ${XCCL_SRC_DIR}/test/start_test.sh ${XCCL_SRC_DIR}/test/torch_bcast_test.py --backend=gloo
