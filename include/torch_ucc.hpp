/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#pragma once

#include <torch/python.h>

#include <exception>
#include <memory>
#include <vector>

#include <pybind11/chrono.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

namespace c10d {

#define TORCH_UCC_DEVICE_NOT_SET -2
#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_TAG_BITS 32
#define TORCH_UCX_OOB_BITS 1

#define TORCH_UCX_COMM_BITS_OFFSET 0
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_TAG_BITS_OFFSET (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS)
#define TORCH_UCX_OOB_BITS_OFFSET \
  (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS + TORCH_UCX_TAG_BITS)

#define TORCH_UCX_MAX_COMM ((((uint64_t)1) << TORCH_UCX_COMM_BITS) - 1)
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_MAX_TAG ((((uint64_t)1) << TORCH_UCX_TAG_BITS) - 1)
#define TORCH_UCX_MAX_OOB ((((uint64_t)1) << TORCH_UCX_OOB_BITS) - 1)

#define TORCH_UCX_COMM_MASK (TORCH_UCX_MAX_COMM << TORCH_UCX_COMM_BITS_OFFSET)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)
#define TORCH_UCX_TAG_MASK (TORCH_UCX_MAX_TAG << TORCH_UCX_TAG_BITS_OFFSET)
#define TORCH_UCX_OOB_MASK (TORCH_UCX_MAX_OOB << TORCH_UCX_OOB_BITS_OFFSET)

#define TORCH_UCX_MAKE_P2P_TAG(_tag, _rank, _comm)       \
  ((((uint64_t)(_tag)) << TORCH_UCX_TAG_BITS_OFFSET) |   \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_comm)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_OOB_TAG(_tag, _rank, _comm)       \
  ((((uint64_t)(_tag)) << TORCH_UCX_OOB_BITS_OFFSET) |   \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_rank)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_SEND_TAG(_ucp_tag, _tag, _rank, _comm)      \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm)); \
  } while (0)

#define TORCH_UCX_ANY_SOURCE (TORCH_UCX_MAX_RANK - 1)
#define TORCH_UCX_ANY_SOURCE_MASK (~TORCH_UCX_RANK_MASK)
#define TORCH_UCX_SPECIFIC_SOURCE_MASK ((uint64_t)-1)

#define TORCH_UCX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank, _comm) \
  do {                                                                       \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm));           \
    if ((_rank) == TORCH_UCX_ANY_SOURCE) {                                   \
      (_ucp_tag_mask) = TORCH_UCX_ANY_SOURCE_MASK;                           \
    } else {                                                                 \
      (_ucp_tag_mask) = TORCH_UCX_SPECIFIC_SOURCE_MASK;                      \
    }                                                                        \
  } while (0)

#define TORCH_UCX_MAKE_OOB_SEND_TAG(_ucp_tag, _tag, _rank, _comm)  \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm)); \
  } while (0)

#define TORCH_UCX_MAKE_OOB_RECV_TAG(                               \
    _ucp_tag, _ucp_tag_mask, _tag, _rank, _comm)                   \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm)); \
    (_ucp_tag_mask) = (uint64_t)-1;                                \
  } while (0)

enum torch_ucx_tag_type_t { TORCH_UCX_P2P_TAG, TORCH_UCX_OOB_TAG };

class CommPG;

class CommBase {
 public:
  CommBase() {}
  virtual void progress() = 0;
  virtual ~CommBase() {}
};

class ProcessGroupUCC : public ProcessGroup {
 public:
  class WorkData {
   public:
    WorkData() {}
    virtual ~WorkData() {}
  };
  class AlltoallWorkData : public WorkData {
   public:
    AlltoallWorkData(int size)
        : send_lengths(size),
          send_offsets(size),
          recv_lengths(size),
          recv_offsets(size) {}
    std::vector<uint32_t> send_lengths;
    std::vector<uint32_t> send_offsets;
    std::vector<uint32_t> recv_lengths;
    std::vector<uint32_t> recv_offsets;
  };

  class AllgatherWorkData : public WorkData {
   public:
    AllgatherWorkData(int size)
        : recv_lengths(size),
          recv_offsets(size) {}
    std::vector<uint64_t> recv_lengths;
    std::vector<uint64_t> recv_offsets;
  };

  class WorkUCC : public ProcessGroup::Work {
    friend class ProcessGroupUCC;
    friend class CommPG;

   public:
    WorkUCC(
        OpType opType,
        ucc_status_t status,
        ucc_coll_req_h request,
        CommBase* comm)
        : ProcessGroup::Work(-1, opType),
          status_(status),
          request_(request),
          comm_(comm) {}
    ~WorkUCC();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    void finalize();
    std::unique_ptr<WorkData> data;

   protected:
    ucc_status_t status_;
    ucc_coll_req_h request_;
    CommBase* comm_;
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1);

  void initComm(c10::Device dev);

  ~ProcessGroupUCC() override;

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  static c10::intrusive_ptr<ProcessGroup> createProcessGroupUCC(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void ProcessGroupUCCConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("ucc", py::cpp_function(createProcessGroupUCC));
  }

 protected:
  c10::intrusive_ptr<Store> store_;
  std::shared_ptr<CommPG> comm;
  uint32_t comm_id;
  std::vector<ucp_ep_h> eps;
  ucc_team_h team;
};

class CommUCX : public CommBase {
 public:
  ucp_context_h context;
  ucp_worker_h worker;

 public:
  void progress() override;
  CommUCX(int comm_size);
  ~CommUCX();
};

class CommUCC : public CommBase {
 public:
  ucc_lib_h lib;
  ucc_context_h context;

 public:
  void progress() override;
  CommUCC(int comm_size);
  ~CommUCC();
};

class CommPG {
  CommUCX ucx_comm;
  CommUCC ucc_comm;
  c10::DeviceIndex device_index;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<c10::intrusive_ptr<ProcessGroupUCC::WorkUCC>> progress_queue;
  bool stop_progress_loop;

 public:
  c10::DeviceIndex cuda_device_index;
  CommPG(int comm_size, c10::Device dev) :
    ucx_comm(comm_size),
    ucc_comm(comm_size),
    cuda_device_index(TORCH_UCC_DEVICE_NOT_SET) {
    if (dev.is_cuda()) {
      cuda_device_index = dev.index();
    }
    stop_progress_loop = false;
    progress_thread = std::thread(&CommPG::progress_loop, this);
    pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
  }
  ~CommPG() {
    std::unique_lock<std::mutex> lock(mutex);
    queue_consume_cv.wait(lock, [&] { return progress_queue.empty(); });
    stop_progress_loop = true;
    lock.unlock();
    queue_produce_cv.notify_all();
    progress_thread.join();
  }

  void ucx_connect_eps(
      std::vector<ucp_ep_h>& eps,
      int rank,
      int size,
      const c10::intrusive_ptr<Store>& store);

  void ucx_disconnect_eps(
      std::vector<ucp_ep_h>& eps,
      const c10::intrusive_ptr<Store>& store);

  void ucc_create_team(
      ucc_team_h& team,
      int rank,
      int size,
      const c10::intrusive_ptr<Store>& store);
  void ucc_destroy_team(ucc_team_h& team);

  c10::intrusive_ptr<ProcessGroup::Work> enqueue_p2p(
      OpType opType,
      ucc_coll_req_h request) {
    if (request == nullptr) {
      // p2p2 request completed immediately don't save it to progress queue
      return c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
          opType, UCC_OK, request, &ucx_comm);
    }
    std::unique_lock<std::mutex> lock(mutex);
    auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
        opType, UCC_INPROGRESS, request, &ucx_comm);
    progress_queue.push_back(work);
    lock.unlock();
    queue_produce_cv.notify_one();
    return work;
  }

  c10::intrusive_ptr<ProcessGroup::Work> enqueue_collective(
      OpType opType,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      ucc_team_h& team) {
    std::unique_lock<std::mutex> lock(mutex);
    ucc_coll_req_h request;
    ucc_status_t st;
    st = ucc_collective_init(&coll, &request, team);
    if (st != UCC_OK) {
      LOG(ERROR) << "failed to init collective: " << ucc_status_string(st);
      throw std::runtime_error(ucc_status_string(st));
    }
    st = ucc_collective_post(request);
    if (st != UCC_OK) {
      LOG(ERROR) << "failed to post collective: " << ucc_status_string(st);
      throw std::runtime_error(ucc_status_string(st));
    }
    auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
        opType, UCC_INPROGRESS, request, &ucc_comm);
    work->data = std::move(data);
    progress_queue.push_back(work);
    lock.unlock();
    queue_produce_cv.notify_one();
    return work;
  }

  static std::shared_ptr<CommPG> get_comm(uint32_t& id, c10::Device dev, int comm_size) {
    static std::mutex m;
    static std::weak_ptr<CommPG> comm;
    static uint32_t comm_id;

    std::lock_guard<std::mutex> lock(m);
    id = (comm_id++ % TORCH_UCX_COMM_BITS);
    std::shared_ptr<CommPG> shared_comm = comm.lock();
    if (!shared_comm) {
      shared_comm = std::make_shared<CommPG>(comm_size, dev);
      comm = shared_comm;
    } else {
      if (dev.is_cuda()) {
        if ((shared_comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
            (shared_comm->cuda_device_index != dev.index())) {
          LOG(ERROR)
              << "ucc communicator was initialized with different cuda device,"
              << "multi device is not supported";
          throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
        }
        shared_comm->cuda_device_index = dev.index();
      }
    }
    return shared_comm;
  }

  void progress_loop() {
    std::unique_lock<std::mutex> lock(mutex);
#ifdef USE_CUDA
    bool device_set = false;
#endif
    while (!stop_progress_loop) {
      if (progress_queue.empty()) {
        queue_produce_cv.wait(lock);
        continue;
      }
      auto work = progress_queue.front();
      progress_queue.pop_front();
      lock.unlock();
      queue_consume_cv.notify_one();
#ifdef USE_CUDA
      if ((!device_set) && (cuda_device_index != TORCH_UCC_DEVICE_NOT_SET)) {
        c10::cuda::set_device(cuda_device_index);
        device_set = true;
      }
#endif
      while (work->request_->status == UCC_INPROGRESS) {
        work->comm_->progress();
      }
      lock.lock();
      work->finalize();
    }
  }
  ucc_coll_req_h send_nb(
      ucp_ep_h ep,
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag);
  ucc_coll_req_h recv_nb(
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag,
      ucp_tag_t ucp_tag_mask);
};

} // namespace c10d
