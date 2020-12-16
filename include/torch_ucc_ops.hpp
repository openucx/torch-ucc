/**
 * * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/Types.hpp>
#include <torch_ucc_sendrecv.hpp>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif
namespace c10d {

enum torch_ucc_status_t {
  TORCH_UCC_OK = 0,
  TORCH_UCC_INPROGRESS = 1,
  TORCH_UCC_OPERATION_INITIALIZED = 2,
  TORCH_UCC_ERROR = -1,
};

struct torch_ucc_coll_comm_t {
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAStream> stream;
#endif
  int dummy;
};

struct torch_ucc_coll_request_t {
  c10::Device device;
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
#ifdef USE_CUDA
  at::cuda::CUDAEvent tensor_ready;
#endif
  torch_ucc_coll_request_t(): device(c10::DeviceType::CPU) {}
};

struct torch_ucc_coll_ops_t {
  torch_ucc_status_t (*coll_comm_init)(
      torch_ucx_comm_t* p2p_comm,
      torch_ucc_coll_comm_t** coll_comm);

  torch_ucc_status_t (*allgather)(
      torch_ucc_coll_comm_t* coll_comm,
      std::vector<at::Tensor>& input_tensor,
      std::vector<at::Tensor>& output_tensors,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*alltoall)(
      torch_ucc_coll_comm_t* coll_comm,
      at::Tensor& input_tensor,
      at::Tensor& output_tensor,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*alltoallv)(
      torch_ucc_coll_comm_t* coll_comm,
      at::Tensor& input_tensor,
      uint32_t* send_lengths,
      uint32_t* send_offsets,
      at::Tensor& output_tensor,
      uint32_t* recv_lengths,
      uint32_t* recv_offsets,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*allreduce)(
      torch_ucc_coll_comm_t* coll_comm,
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*barrier)(
      torch_ucc_coll_comm_t* coll_comm,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*broadcast)(
      torch_ucc_coll_comm_t* coll_comm,
      std::vector<at::Tensor>& tensors,
      int root,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*coll_progress)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_test)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_finalize)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_comm_close)(torch_ucc_coll_comm_t* coll_comm);
};

extern torch_ucc_coll_ops_t xccl_coll_ops;

inline void torch_ucc_coll_request_init(
    torch_ucc_coll_comm_t* coll_comm,
    torch_ucc_coll_request_t* request,
    std::vector<at::Tensor>* srcPtr,
    std::vector<at::Tensor>* dstPtr) {
  if (srcPtr) {
    request->src = *srcPtr;
    request->device = request->src[0].device();
#ifdef USE_CUDA
    if (request->device.is_cuda()) {
      request->tensor_ready.record(
          at::cuda::getCurrentCUDAStream(request->device.index()));
      if (coll_comm->stream == nullptr) {
        coll_comm->stream = std::make_unique<at::cuda::CUDAStream>(
            at::cuda::getStreamFromPool(request->device.index()));
      }
      request->tensor_ready.block(*coll_comm->stream);
    }
#endif
  }

  if (dstPtr) {
    request->dst = *dstPtr;
    if (request->src[0].device() != request->dst[0].device()) {
      fprintf(stderr, "ProcessGroupUCC: multidevice is not supported\n");
    }
  }
}

inline torch_ucc_status_t torch_ucc_coll_ops_init(
    torch_ucc_coll_ops_t* coll_ops) {
  *coll_ops = xccl_coll_ops;
  return TORCH_UCC_OK;
}

}; // namespace c10d
