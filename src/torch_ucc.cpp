/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include <torch_ucc.hpp>
#include <stdio.h>
#include <torch_ucc_sendrecv.hpp>

#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#endif

namespace c10d {

void ProcessGroupUCC::check_tensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error("ProcessGroupUCC takes 1 tensor");
  }
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error(
        "ProcessGroupUCC input tensor has to be contiguous");
  }
  if (tensors[0].is_sparse()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be dense");
  }
  // TODO: check cuda case
}

static torch_ucc_status_t compute_lengths_offsets(
    int group_size,
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    uint32_t* lengths,
    uint32_t* offsets) {
  bool equal_splits = false;
  size_t dim0_size = tensor.size(0);
  size_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  size_t split_size = 0;
  size_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }

  for (int i = 0; i < group_size; i++) {
    size_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    if ((length > INT_MAX) || (offset > INT_MAX)) {
      return TORCH_UCC_ERROR;
    }
    lengths[i] = length;
    offsets[i] = offset;
    offset += length;
  }

  return TORCH_UCC_OK;
}

ProcessGroupUCC::WorkUCX::~WorkUCX() {
  if (req != NULL) {
    torch_ucx_request_free(req);
  }
}

bool ProcessGroupUCC::WorkUCX::isCompleted() {
  torch_ucx_status_t st;

  st = torch_ucx_req_test(comm, &req, 1, NULL, 1, 1);
  return (st != TORCH_UCX_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCX::isSuccess() const {
  // TODO
  return true;
}

bool ProcessGroupUCC::WorkUCX::wait() {
  return wait(kUnsetTimeout);
}

bool ProcessGroupUCC::WorkUCX::wait(std::chrono::milliseconds timeout) {
  torch_ucx_req_test(comm, &req, 1, NULL, -1, 1);
  return true;
}

ProcessGroupUCC::WorkColl::~WorkColl() {
  if (coll_req != NULL) {
    coll_ops.coll_finalize(coll_req);
  }

  if (alltoall_len_offset != NULL) {
    delete[] alltoall_len_offset;
  }
}

bool ProcessGroupUCC::WorkColl::isCompleted() {
  torch_ucc_status_t st;

  if (!no_progress) {
    coll_ops.coll_progress(coll_req);
  }
  st = coll_ops.coll_test(coll_req);

  return (st != TORCH_UCC_INPROGRESS);
}

bool ProcessGroupUCC::WorkColl::isSuccess() const {
  // TODO
  return true;
}

bool ProcessGroupUCC::WorkColl::wait() {
  return wait(kUnsetTimeout);
}

bool ProcessGroupUCC::WorkColl::wait(std::chrono::milliseconds timeout) {
  while (!isCompleted()) {
  };

  return true;
}

void ProcessGroupUCC::read_config() {
  char* env;

  config.enable_progress_thread = true;
  env = std::getenv("TORCH_UCC_THREAD_ENABLE");
  if (env) {
    config.enable_progress_thread = std::atoi(env);
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const std::shared_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store), stop_progress_loop(false) {
  torch_ucx_status_t st;
  torch_ucc_status_t st_ucc;

  read_config();
  st = torch_ucx_comm_init(&ucx_comm, size, rank, store_);
  if (st != TORCH_UCX_OK) {
    throw std::runtime_error("ProcessGroupUCC init failed");
  }

  st_ucc = torch_ucc_coll_ops_init(&coll_ops);
  if (st_ucc != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC failed to init collops");
  }

  st_ucc = coll_ops.coll_comm_init(ucx_comm, &coll_comm);
  if (st_ucc != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC failed to init collective comm");
  }

  if (config.enable_progress_thread) {
    progress_thread = std::thread(&ProcessGroupUCC::progress_loop, this);
  }
}

void ProcessGroupUCC::progress_loop() {
  std::unique_lock<std::mutex> lock(pg_mutex);
  torch_ucc_status_t st;
  torch_ucc_coll_request_t* req;

#ifdef USE_CUDA
  auto device = c10::Device(c10::DeviceType::CUDA, (c10::DeviceIndex)0);
  at::cuda::OptionalCUDAGuard guard(device);
  cudaSetDevice(0);
#endif

  while (!stop_progress_loop) {
    if (progress_queue.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    req = progress_queue.front();
    progress_queue.pop_front();
    lock.unlock();
    queue_consume_cv.notify_one();
#ifdef USE_CUDA
    if (req->dev_type == c10::DeviceType::CUDA) {
      guard.set_index(req->dev_index);
    }
#endif
    do {
      st = coll_ops.coll_progress(req);
    } while ((coll_ops.coll_test(req) == TORCH_UCC_INPROGRESS) ||
             (st != TORCH_UCC_OK));
    if (st != TORCH_UCC_OK) {
      fprintf(stderr, "ProcessGroupUCC: coll progress failed\n");
    }
    lock.lock();
  }
}

void ProcessGroupUCC::enqueue_request(torch_ucc_coll_request_t* req) {
  std::unique_lock<std::mutex> lock(pg_mutex);
  progress_queue.push_back(req);
  lock.unlock();
  queue_produce_cv.notify_one();
}

ProcessGroupUCC::~ProcessGroupUCC() {
  if (config.enable_progress_thread) {
    std::unique_lock<std::mutex> lock(pg_mutex);
    queue_consume_cv.wait(lock, [&] { return progress_queue.empty(); });
    stop_progress_loop = true;
    lock.unlock();
    queue_produce_cv.notify_all();
    progress_thread.join();
  }

  coll_ops.coll_comm_close(coll_comm);
  torch_ucx_comm_close(ucx_comm, store_);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  auto request = std::make_shared<ProcessGroupUCC::WorkColl>(coll_ops);
  auto& tensor = tensors[0];
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  st = coll_ops.broadcast(coll_comm, tensor, opts.rootRank, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: broadcast failed");
  }
  request->coll_req = coll_req;
  if (config.enable_progress_thread) {
    request->coll_req->dev_index = tensor.device().index();
    request->coll_req->dev_type = tensor.device().type();
    enqueue_request(request->coll_req);
    request->no_progress = true;
  }

  return request;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  auto request = std::make_shared<ProcessGroupUCC::WorkColl>(coll_ops);
  auto& tensor = tensors[0];
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  st = coll_ops.allreduce(coll_comm, tensor, opts, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: allreduce failed");
  }
  request->coll_req = coll_req;
  if (config.enable_progress_thread) {
    request->coll_req->dev_index = tensor.device().index();
    request->coll_req->dev_type = tensor.device().type();
    enqueue_request(request->coll_req);
    request->no_progress = true;
  }

  return request;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  auto request = std::make_shared<ProcessGroupUCC::WorkColl>(coll_ops);
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  st = coll_ops.allgather(
      coll_comm, inputTensors[0], outputTensors[0], &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: allgather failed");
  }
  request->coll_req = coll_req;
  if (config.enable_progress_thread) {
    enqueue_request(request->coll_req);
    request->no_progress = true;
  }
  return request;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& opts) {
  auto request = std::make_shared<ProcessGroupUCC::WorkColl>(coll_ops);
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  st = coll_ops.barrier(coll_comm, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: barrier failed");
  }
  request->coll_req = coll_req;
  if (config.enable_progress_thread) {
    enqueue_request(request->coll_req);
    request->no_progress = true;
  }
  return request;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  auto request = std::make_shared<ProcessGroupUCC::WorkColl>(coll_ops);
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    st = coll_ops.alltoall(coll_comm, inputTensor, outputTensor, &coll_req);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoall_base failed");
    }
  } else {
    request->alltoall_len_offset = new uint32_t[4 * size_];
    uint32_t* send_lengths = request->alltoall_len_offset;
    uint32_t* recv_lengths =
        (uint32_t*)((ptrdiff_t)send_lengths + 1 * size_ * sizeof(uint32_t));
    uint32_t* send_offsets =
        (uint32_t*)((ptrdiff_t)send_lengths + 2 * size_ * sizeof(uint32_t));
    uint32_t* recv_offsets =
        (uint32_t*)((ptrdiff_t)send_lengths + 3 * size_ * sizeof(uint32_t));

    st = compute_lengths_offsets(
        size_, outputSplitSizes, outputTensor, recv_lengths, recv_offsets);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoallv failed");
    }

    st = compute_lengths_offsets(
        size_, inputSplitSizes, inputTensor, send_lengths, send_offsets);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoallv failed");
    }

    coll_ops.alltoallv(
        coll_comm,
        inputTensor,
        send_lengths,
        send_offsets,
        outputTensor,
        recv_lengths,
        recv_offsets,
        &coll_req);
  }
  request->coll_req = coll_req;
  if (config.enable_progress_thread) {
    request->coll_req->dev_index = inputTensor.device().index();
    request->coll_req->dev_type = inputTensor.device().type();
    enqueue_request(request->coll_req);
    request->no_progress = true;
  }
  return request;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  // TODO: check tensor count and type, assume single dense tensor
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucx_status_t st;

  st = torch_ucx_send_nb(
      ucx_comm, tensor.data_ptr(), size, dstRank, tag, &req, TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to send msg");
  }

  return std::make_shared<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucx_status_t st;

  st = torch_ucx_recv_nb(
      ucx_comm, tensor.data_ptr(), size, srcRank, tag, &req, TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to recv msg");
  }

  return std::make_shared<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("ProcessGroupUCC: recvAnysource is not supported");
}

std::shared_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const std::shared_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return std::make_shared<ProcessGroupUCC>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &ProcessGroupUCC::createProcessGroupUCC);
}

} // namespace c10d
