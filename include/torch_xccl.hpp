/**
 * * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <api/xccl.h>
#include <torch_ucc_ops.hpp>
#include <torch_ucc_sendrecv.hpp>

namespace c10d {

struct torch_xccl_comm_t {
  torch_ucc_coll_comm_t super;
  torch_ucx_comm_t* p2p_comm{};
  xccl_lib_h xccl_lib{};
  xccl_context_h xccl_ctx{};
  xccl_team_h xccl_team{};
};

struct torch_xccl_request_t {
  torch_ucc_coll_request_t super;
  xccl_coll_req_h request{};
  xccl_collective_type_t coll_type;
  torch_ucc_status_t status;
  at::Tensor flat_tensor;
#ifdef USE_CUDA
  cudaStream_t stream;
#endif
};

torch_ucc_status_t torch_xccl_comm_init(
    torch_ucx_comm_t* p2p_comm,
    torch_ucc_coll_config_t* coll_config,
    torch_ucc_coll_comm_t** comm);

torch_ucc_status_t torch_xccl_comm_close(torch_ucc_coll_comm_t* comm);

} // namespace c10d
