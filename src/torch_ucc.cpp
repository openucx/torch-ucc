/**
 * * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include <torch_ucc.hpp>
#include <torch_ucc_sendrecv.hpp>
#include <utility>
#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cstdio>

namespace c10d {

const std::map<c10::DeviceType, ucs_memory_type_t> ucs_mtype_map = {
    {c10::kCPU, UCS_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCS_MEMORY_TYPE_CUDA},
    {c10::kHIP, UCS_MEMORY_TYPE_ROCM},
    {c10::kFPGA, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kMSNPU, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kXLA, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kVulkan, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kMetal, UCS_MEMORY_TYPE_UNKNOWN},
};

const std::map<torch_ucc_collective_type_t, c10d::OpType> optype_map = {
    {TORCH_UCC_BARRIER, c10d::OpType::BARRIER},
    {TORCH_UCC_BCAST, c10d::OpType::BROADCAST},
    {TORCH_UCC_ALLREDUCE, c10d::OpType::ALLREDUCE},
    {TORCH_UCC_ALLTOALL, c10d::OpType::ALLTOALL_BASE},
    {TORCH_UCC_ALLTOALLV, c10d::OpType::ALLTOALL_BASE},
    {TORCH_UCC_ALLGATHER, c10d::OpType::ALLGATHER_BASE},
};

const std::map<torch_ucc_collective_type_t, const char*> torch_ucc_collective_name = {
    {TORCH_UCC_BARRIER, "ucc barrier"},
    {TORCH_UCC_BCAST, "ucc bcast"},
    {TORCH_UCC_ALLREDUCE, "ucc allreduce"},
    {TORCH_UCC_ALLTOALL, "ucc alltoall"},
    {TORCH_UCC_ALLTOALLV, "ucc alltoallv"},
    {TORCH_UCC_ALLGATHER, "ucc allgather"},
};

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
  if (req != nullptr) {
    torch_ucx_request_free(req);
  }
}

bool ProcessGroupUCC::WorkUCX::isCompleted() {
  torch_ucc_status_t st;

  st = torch_ucx_req_test(comm, &req, 1, nullptr, 1, 1);
  return (st != TORCH_UCC_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCX::isSuccess() const {
  // TODO
  return true;
}

bool ProcessGroupUCC::WorkUCX::wait(
  std::chrono::milliseconds /* unused */) {
  torch_ucx_req_test(comm, &req, 1, nullptr, -1, 1);
  return true;
}

ProcessGroupUCC::WorkColl::~WorkColl() {
  if (coll_req != nullptr) {
    if (coll_ops.coll_test(coll_req) != TORCH_UCC_OK) {
      fprintf(
          stderr,
          "ProcessGroupUCC: warn removing request before collective finish\n");
    }
    coll_ops.coll_finalize(coll_req);
  }

  if (scratch != nullptr) {
    delete[] scratch;
  }
}

bool ProcessGroupUCC::WorkColl::isCompleted() {
  torch_ucc_status_t st;

  if (!external_progress) {
    coll_ops.coll_progress(coll_req);
    st = coll_ops.coll_test(coll_req);
    if (st != TORCH_UCC_INPROGRESS) {
      work_list.erase(work_list_entry);
    }
  } else {
    st = coll_ops.coll_test(coll_req);
  }
  return (st != TORCH_UCC_INPROGRESS);
}

bool ProcessGroupUCC::WorkColl::isSuccess() const {
  // TODO
  return true;
}

bool ProcessGroupUCC::WorkColl::wait(
  std::chrono::milliseconds /* unused */) {
  if (blocking_wait || !coll_req->device.is_cuda()) {
    while (!isCompleted()) {
    };
  } else {
    coll_ops.coll_fence(coll_req);
  }
  return true;
}

void ProcessGroupUCC::read_config() {
  char* env;

  config.enable_progress_thread = true;
  env = std::getenv("TORCH_UCC_THREAD_ENABLE");
  if (env) {
    config.enable_progress_thread = std::atoi(env);
  }
  config.gpu_barrier = true;
  env = std::getenv("TORCH_UCC_GPU_BARRIER");
  if (env) {
    config.gpu_barrier = std::atoi(env);
  }
  config.enable_profiling = false;
  env = std::getenv("TORCH_UCC_PROFILING_ENABLE");
  if (env) {
    config.enable_profiling = std::atoi(env);
  }
  config.serialize = false;
  env = std::getenv("TORCH_UCC_SERIALIZE_COLL");
  if (env) {
    config.serialize = std::atoi(env);
  }
  for (int i = 0; i < TORCH_UCC_COLL_LAST; i++) {
    config.blocking_wait[i] = true;
  }
  env = std::getenv("TORCH_UCC_ALLGATHER_BLOCKING_WAIT");
  if (env) {
    config.blocking_wait[TORCH_UCC_ALLGATHER] = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_ALLREDUCE_BLOCKING_WAIT");
  if (env) {
    config.blocking_wait[TORCH_UCC_ALLREDUCE] = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_ALLTOALL_BLOCKING_WAIT");
  if (env) {
    config.blocking_wait[TORCH_UCC_ALLTOALL] = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_ALLTOALLV_BLOCKING_WAIT");
  if (env) {
    config.blocking_wait[TORCH_UCC_ALLTOALLV] = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_BCAST_BLOCKING_WAIT");
  if (env) {
    config.blocking_wait[TORCH_UCC_BCAST] = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_BARRIER_BLOCKING_WAIT");
  if (env) {
    config.blocking_wait[TORCH_UCC_BARRIER] = std::atoi(env);
  }

  config.high_priority_stream = false;
  env = std::getenv("TORCH_UCC_HIGH_PRIORITY_STREAM");
  if (env) {
    config.high_priority_stream = std::atoi(env);
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store), stop_progress_loop(false) {
  torch_ucc_status_t st;

  read_config();
  st = torch_ucc_coll_ops_init(&coll_ops);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC failed to init collops");
  }
  coll_comm = nullptr;
  ucx_comm = nullptr;
}

void ProcessGroupUCC::start_progress_thread() {
  if (config.enable_progress_thread) {
    c10::DeviceIndex dev_idx = 0;
#ifdef USE_CUDA
    dev_idx = c10::cuda::current_device();
#endif
    progress_thread = std::thread(&ProcessGroupUCC::progress_loop, this, dev_idx);
  }
}

torch_ucc_coll_comm_t* ProcessGroupUCC::get_coll_comm() {
  if (coll_comm == nullptr) {
    torch_ucc_status_t st;
    torch_ucc_coll_config_t cfg;

    get_p2p_comm();
    memcpy(cfg.blocking_wait, config.blocking_wait, sizeof(cfg.blocking_wait));
    cfg.high_priority_stream = config.high_priority_stream;
    cfg.gpu_barrier = config.gpu_barrier;
    cfg.serialize = config.serialize;
    st = coll_ops.coll_comm_init(ucx_comm, &cfg, &coll_comm);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error(
          "ProcessGroupUCC failed to init collective comm");
    }
    start_progress_thread();
  }

  return coll_comm;
}

torch_ucx_comm_t* ProcessGroupUCC::get_p2p_comm() {
  if (ucx_comm == nullptr) {
    torch_ucc_status_t st;

    st = torch_ucx_comm_init(&ucx_comm, size_, rank_, store_);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC init failed");
    }
  }

  return ucx_comm;
}

void ProcessGroupUCC::progress_loop(c10::DeviceIndex default_dev_idx) {
  std::unique_lock<std::mutex> lock(pg_mutex);
  torch_ucc_status_t st;
#ifdef USE_CUDA
  at::cuda::OptionalCUDAGuard guard(default_dev_idx);
  if (default_dev_idx == 0) {
    c10::cuda::set_device(default_dev_idx);
  }
#endif
  while (!stop_progress_loop) {
    if (progress_list.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    auto work_coll = progress_list.front();
    progress_list.pop_front();
    lock.unlock();
    queue_consume_cv.notify_one();
#ifdef USE_CUDA
    if (work_coll->coll_req->device.is_cuda()) {
      guard.set_device(work_coll->coll_req->device);
    }
#endif
    do {
      st = coll_ops.coll_progress(work_coll->coll_req);
    } while (
        (coll_ops.coll_test(work_coll->coll_req) == TORCH_UCC_INPROGRESS) &&
        (st == TORCH_UCC_OK));
    if (st != TORCH_UCC_OK) {
      fprintf(stderr, "ProcessGroupUCC: coll progress failed\n");
    }
    if (config.enable_profiling) {
      work_coll->finish();
    }
    lock.lock();
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::enqueue_request(
    torch_ucc_coll_request_t* req,
    void* scratch) {
  std::unique_lock<std::mutex> lock(pg_mutex);

  auto iter = progress_list.emplace(
      progress_list.end(),
      c10::make_intrusive<ProcessGroupUCC::WorkColl>(
        coll_ops,
        progress_list,
        rank_,
        optype_map.at(req->coll_type),
        config.enable_profiling ? torch_ucc_collective_name.at(req->coll_type): nullptr));
  (*iter)->work_list_entry = iter;
  (*iter)->coll_req = req;
  (*iter)->blocking_wait = config.blocking_wait[req->coll_type];
  (*iter)->external_progress = config.enable_progress_thread;
  (*iter)->scratch = (char*)scratch;
  auto workreq = (*iter);
  lock.unlock();
  queue_produce_cv.notify_one();
  return workreq;
}

ProcessGroupUCC::~ProcessGroupUCC() {
  if (config.enable_progress_thread) {
    std::unique_lock<std::mutex> lock(pg_mutex);
    queue_consume_cv.wait(lock, [&] { return progress_list.empty(); });
    stop_progress_loop = true;
    lock.unlock();
    queue_produce_cv.notify_all();
    progress_thread.join();
  }
  if (progress_list.size() != 0) {
    fprintf(stderr, "ProcessGroupUCC: warnning progress list is not empty\n");
  }
  if (coll_comm != nullptr) {
    coll_ops.coll_comm_close(coll_comm);
  }
  if (ucx_comm != nullptr) {
    torch_ucx_comm_close(ucx_comm, store_);
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  check_tensor(tensors);
  c10::DeviceGuard guard(tensors[0].device());
  ucc_comm = get_coll_comm();
  st = coll_ops.broadcast(ucc_comm, tensors, opts.rootRank, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: broadcast failed");
  }
  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  check_tensor(tensors);
  c10::DeviceGuard guard(tensors[0].device());
  ucc_comm = get_coll_comm();
  st = coll_ops.allreduce(ucc_comm, tensors, opts, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: allreduce failed");
  }

  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  check_tensor(inputTensors);
  c10::DeviceGuard guard(inputTensors[0].device());
  ucc_comm = get_coll_comm();
  st = coll_ops.allgather(ucc_comm, inputTensors, outputTensors[0], &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: allgather failed");
  }
  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& /* unused */) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;

  ucc_comm = get_coll_comm();
  st = coll_ops.barrier(ucc_comm, &coll_req);
  if (st != TORCH_UCC_OK) {
    throw std::runtime_error("ProcessGroupUCC: barrier failed");
  }
  return enqueue_request(coll_req, nullptr);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  torch_ucc_coll_comm_t* ucc_comm;
  torch_ucc_coll_request_t* coll_req;
  torch_ucc_status_t st;
  uint32_t* scratch;

  c10::DeviceGuard guard(inputTensor.device());
  ucc_comm = get_coll_comm();
  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    scratch = nullptr;
    st = coll_ops.alltoall(ucc_comm, inputTensor, outputTensor, &coll_req);
    if (st != TORCH_UCC_OK) {
      throw std::runtime_error("ProcessGroupUCC: alltoall_base failed");
    }
  } else {
    scratch = new uint32_t[4 * size_];
    uint32_t* send_lengths = scratch;
    uint32_t* recv_lengths = scratch + 1 * size_;
    uint32_t* send_offsets = scratch + 2 * size_;
    uint32_t* recv_offsets = scratch + 3 * size_;
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
        ucc_comm,
        inputTensor,
        send_lengths,
        send_offsets,
        outputTensor,
        recv_lengths,
        recv_offsets,
        &coll_req);
  }
  return enqueue_request(coll_req, scratch);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucc_status_t st;
  torch_ucx_comm_t* p2p_comm;

  p2p_comm = get_p2p_comm();
  st = torch_ucx_send_nb(
      p2p_comm,
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      size,
      dstRank,
      tag,
      &req,
      TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to send msg");
  }

  return c10::make_intrusive<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucc_status_t st;
  torch_ucx_comm_t* p2p_comm;

  p2p_comm = get_p2p_comm();
  st = torch_ucx_recv_nb(
      p2p_comm,
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      size,
      srcRank,
      tag,
      &req,
      TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to recv msg");
  }

  return c10::make_intrusive<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  torch_ucx_request_t* req;
  torch_ucc_status_t st;
  torch_ucx_comm_t* p2p_comm;

  p2p_comm = get_p2p_comm();
  st = torch_ucx_recv_nb(
      p2p_comm,
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      size,
      TORCH_UCX_ANY_SOURCE,
      tag,
      &req,
      TORCH_UCX_P2P_TAG);
  if (st < 0) {
    throw std::runtime_error("ProcessGroupUCC: failed to recv msg");
  }

  return c10::make_intrusive<ProcessGroupUCC::WorkUCX>(req, ucx_comm);
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &ProcessGroupUCC::createProcessGroupUCC);
}

} // namespace c10d
