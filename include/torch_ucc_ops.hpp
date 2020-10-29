/**
 * * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/Types.hpp>
#include <torch_ucc_sendrecv.hpp>
#include <torch_ucc_status.hpp>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif
namespace c10d {

enum torch_ucx_memtype_t { TORCH_UCX_HOST, TORCH_UCX_CUDA };

struct torch_ucc_coll_comm_t {
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAStream> stream;
#endif
  int dummy;
};

struct torch_ucc_coll_request_t {
  c10::DeviceIndex dev_index;
  c10::DeviceType dev_type;
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
#ifdef USE_CUDA
  at::cuda::CUDAEvent tensor_ready;
#endif
};

struct torch_ucc_coll_ops_t {
  torch_ucc_status_t (*coll_comm_init)(
      torch_ucx_comm_t* p2p_comm,
      torch_ucc_coll_comm_t** coll_comm);

  torch_ucc_status_t (*allgather)(
      torch_ucc_coll_comm_t* coll_comm,
      at::Tensor& input_tensor,
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
      at::Tensor& tensor,
      const AllreduceOptions& opts,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*barrier)(
      torch_ucc_coll_comm_t* coll_comm,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*broadcast)(
      torch_ucc_coll_comm_t* coll_comm,
      at::Tensor& tensor,
      int root,
      torch_ucc_coll_request_t** request);

  torch_ucc_status_t (*coll_progress)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_test)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_finalize)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_comm_close)(torch_ucc_coll_comm_t* coll_comm);
};

extern torch_ucc_coll_ops_t ucx_coll_ops;

#ifdef WITH_XCCL
extern torch_ucc_coll_ops_t xccl_coll_ops;
#endif

inline void torch_ucc_coll_request_init(
    torch_ucc_coll_comm_t* coll_comm,
    torch_ucc_coll_request_t* request,
    std::vector<at::Tensor>* srcPtr,
    std::vector<at::Tensor>* dstPtr) {
  if (srcPtr) {
    request->src = *srcPtr;
#ifdef USE_CUDA
    if (request->src[0].is_cuda()) {
      c10::DeviceIndex dev_index = request->src[0].device().index();

      request->tensor_ready.record(at::cuda::getCurrentCUDAStream(dev_index));
      if (coll_comm->stream == NULL) {
        coll_comm->stream = std::make_unique<at::cuda::CUDAStream>(
            at::cuda::getStreamFromPool(dev_index));
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
  char* env;

  env = std::getenv("TORCH_UCC_COLL_BACKEND");
  if ((env != NULL) && (!strcasecmp(env, "xccl"))) {
#ifdef WITH_XCCL
    *coll_ops = xccl_coll_ops;
#else
    fprintf(
        stderr, "ProcessGroupUCC: plugin wasn't compiled with XCCL support\n");
    return TORCH_UCC_ERROR;
#endif
  } else {
    *coll_ops = ucx_coll_ops;
  }

  return TORCH_UCC_OK;
}

}; // namespace c10d
