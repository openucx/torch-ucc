/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <torch_ucc_status.hpp>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#endif

#include <ATen/ATen.h>
#include <torch_ucc_ops.hpp>
#include <torch_ucc_status.hpp>
#include "torch_ucc_sendrecv.hpp"
namespace c10d {

struct torch_ucx_coll_request_t;
typedef torch_ucc_status_t (*torch_ucx_progress_p)(
    torch_ucx_coll_request_t* request);

struct torch_ucx_coll_config_t {
  int chunk;
  bool reverse;
  int max_polls;
};

struct torch_ucx_coll_comm_t {
  torch_ucc_coll_comm_t super;
  torch_ucx_comm_t* p2p_comm;
  torch_ucx_coll_config_t config;
  uint32_t last_tag;
};

struct torch_ucx_coll_request_t {
  torch_ucc_coll_request_t super;
  torch_ucx_coll_comm_t* comm;
  uint32_t tag;
  torch_ucx_progress_p progress;
  torch_ucc_status_t status;
  torch_ucx_memtype_t src_buf_mtype;
  void* src_buffer;
  torch_ucx_memtype_t dst_buf_mtype;
  void* dst_buffer;
  size_t len;
  at::ScalarType send_data_type;
  uint32_t* send_lengths;
  uint32_t* send_offsets;
  at::ScalarType recv_data_type;
  uint32_t* recv_lengths;
  uint32_t* recv_offsets;
  torch_ucx_request_t** reqs;
  int n_sreqs;
  int n_rreqs;
#ifdef USE_CUDA
  at::cuda::CUDAEvent memcpy_done;
#endif
};

torch_ucc_status_t torch_ucx_alltoall(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& input_tensor,
    at::Tensor& output_tensor,
    torch_ucc_coll_request_t** request);

torch_ucc_status_t torch_ucx_alltoallv(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& input_tensor,
    uint32_t* send_lengths,
    uint32_t* send_offsets,
    at::Tensor& output_tensor,
    uint32_t* recv_lengths,
    uint32_t* recv_offsets,
    torch_ucc_coll_request_t** request);

} // namespace c10d
