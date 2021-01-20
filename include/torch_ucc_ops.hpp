/**
 * * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/Types.hpp>
#include <torch_ucc_sendrecv.hpp>
#include <queue>
#include <mutex>
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

typedef enum {
  TORCH_UCC_BARRIER = 0,
  TORCH_UCC_BCAST,
  TORCH_UCC_ALLREDUCE,
  TORCH_UCC_ALLTOALL,
  TORCH_UCC_ALLTOALLV,
  TORCH_UCC_ALLGATHER,
  TORCH_UCC_COLL_LAST,
} torch_ucc_collective_type_t;

struct torch_ucc_coll_config_t {
  bool blocking_wait[TORCH_UCC_COLL_LAST];
  bool high_priority_stream;
};

struct torch_ucc_coll_comm_t {
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAStream> stream;
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>> event_pool;
  std::mutex event_pool_mutex;
#endif
  torch_ucc_coll_config_t config;
};

struct torch_ucc_coll_request_t {
  torch_ucc_coll_comm_t *coll_comm;
  c10::Device device;
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
  torch_ucc_collective_type_t coll_type;
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAEvent> tnsr_ready;
  std::unique_ptr<at::cuda::CUDAEvent> coll_finished;
#endif
  torch_ucc_coll_request_t(): device(c10::DeviceType::CPU) {}
  ~torch_ucc_coll_request_t() {
#ifdef USE_CUDA
    if (device.is_cuda()) {
      std::lock_guard<std::mutex> lock(coll_comm->event_pool_mutex);
      coll_comm->event_pool.push(std::move(tnsr_ready));
      coll_comm->event_pool.push(std::move(coll_finished));
    }
#endif
  }
};

struct torch_ucc_coll_ops_t {
  torch_ucc_status_t (*coll_comm_init)(
      torch_ucx_comm_t* p2p_comm,
      torch_ucc_coll_config_t* coll_config,
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

  torch_ucc_status_t (*coll_fence)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_finalize)(torch_ucc_coll_request_t* request);

  torch_ucc_status_t (*coll_comm_close)(torch_ucc_coll_comm_t* coll_comm);
};

extern torch_ucc_coll_ops_t xccl_coll_ops;

inline void torch_ucc_coll_request_init(
    torch_ucc_coll_comm_t* coll_comm,
    torch_ucc_collective_type_t coll_type,
    torch_ucc_coll_request_t* request,
    std::vector<at::Tensor>* srcPtr,
    std::vector<at::Tensor>* dstPtr) {
  request->coll_comm = coll_comm;
  request->coll_type = coll_type;
  if (srcPtr) {
    request->src = *srcPtr;
    request->device = request->src[0].device();
#ifdef USE_CUDA
    request->tnsr_ready = nullptr;
    request->coll_finished = nullptr;
    if (request->device.is_cuda()) {
      std::lock_guard<std::mutex> lock(coll_comm->event_pool_mutex);
      if (coll_comm->stream == nullptr) {
        coll_comm->stream = std::make_unique<at::cuda::CUDAStream>(
            at::cuda::getStreamFromPool(coll_comm->config.high_priority_stream,
                                        request->device.index()));
      }
      if (coll_comm->event_pool.empty()) {
        request->tnsr_ready = std::make_unique<at::cuda::CUDAEvent>();
        request->coll_finished = std::make_unique<at::cuda::CUDAEvent>();
      } else {
        request->tnsr_ready = std::move(coll_comm->event_pool.front());
        coll_comm->event_pool.pop();
        request->coll_finished = std::move(coll_comm->event_pool.front());
        coll_comm->event_pool.pop();
      }
      request->tnsr_ready->record(
          at::cuda::getCurrentCUDAStream(request->device.index()));
      request->tnsr_ready->block(*coll_comm->stream);

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
