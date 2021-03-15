#!/bin/bash -eEx
set -o pipefail

command -v mpirun
export UCX_SOCKADDR_CM_ENABLE=n
export UCX_WARN_UNUSED_ENV_VARS=n
MPI_ARGS_COMMON="--allow-run-as-root --oversubscribe -np 8 -H localhost:8 --bind-to none --mca coll ^hcoll --mca btl ^openib --mca mtl ^ofi"
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=hier ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_allreduce
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=hier ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_bcast
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=hier ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_barrier

mpirun ${MPI_ARGS_COMMON} -x XCCL_TEAM_HIER_NODE_LEADER_RANK_ID=3 -x XCCL_TEST_TLS=hier ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_allreduce
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEAM_HIER_NODE_LEADER_RANK_ID=4 -x XCCL_TEST_TLS=hier ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_bcast
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEAM_HIER_NODE_LEADER_RANK_ID=5 -x XCCL_TEST_TLS=hier ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_barrier

mpirun ${MPI_ARGS_COMMON} -x XCCL_TEAM_UCX_ALLREDUCE_ALG_ID=0 -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_allreduce
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEAM_UCX_ALLREDUCE_ALG_ID=1 -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_allreduce
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_bcast
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_barrier
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_alltoall
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_alltoallv
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_allgather
mpirun -x XCCL_TEAM_UCX_ALLTOALL_PAIRWISE_CHUNK=0 ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_alltoall
mpirun -x XCCL_TEAM_UCX_ALLTOALL_PAIRWISE_CHUNK=0 ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=ucx ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_alltoallv
mpirun ${MPI_ARGS_COMMON} -x XCCL_TEST_TLS=hier -x XCCL_TEST_ITERS=500 -x XCCL_TEST_NTHREADS=4 -x XCCL_TEST_CHECK=1 ${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}/test/test_mpi_mt
