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
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>
#include "torch_ucc_sendrecv.hpp"

namespace c10d {

class CommPG;

class ProcessGroupUCC : public ProcessGroup {
 public:
  class WorkUCX : public ProcessGroup::Work {
    friend class ProcessGroupUCC;

   public:
    WorkUCX(torch_ucx_request_t* request) : request_(request) {}
    ~WorkUCX() override;
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

   protected:
    torch_ucx_request_t* request_;
  };
  class WorkUCC : public ProcessGroup::Work {
    friend class ProcessGroupUCC;
    friend class CommUCC;

   public:
    WorkUCC(ucc_coll_req_h request) : request_(request) {}
    ~WorkUCC() override;
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

   protected:
    ucc_coll_req_h request_;
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1);

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

class CommUCX {
 public:
  ucp_context_h context;
  ucp_worker_h worker;

 public:
  CommUCX();
  ~CommUCX();
  torch_ucx_request_t* send_nb(
      ucp_ep_h ep,
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag);
  torch_ucx_request_t* recv_nb(
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag,
      ucp_tag_t ucp_tag_mask);
};

class CommUCC {
 public:
  ucc_lib_h lib;
  ucc_context_h context;

 public:
  CommUCC();
  ~CommUCC();
};

class CommPG {
  CommUCX ucx_comm;
  CommUCC ucc_comm;

  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::list<c10::intrusive_ptr<ProcessGroup::Work>> progress_list;
  bool stop_progress_loop;

 public:
  CommPG() {
    stop_progress_loop = false;
    progress_thread = std::thread(&CommPG::progress_loop, this);
  }
  ~CommPG() {
    std::unique_lock<std::mutex> lock(mutex);
    queue_consume_cv.wait(lock, [&] { return progress_list.empty(); });
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

  void enqueue_request(const c10::intrusive_ptr<ProcessGroup::Work>& work) {
    std::unique_lock<std::mutex> lock(mutex);
    progress_list.push_back(work);
    lock.unlock();
    queue_produce_cv.notify_one();
  }

  static std::shared_ptr<CommPG> get_comm(uint32_t& id) {
    static std::mutex m;
    static std::weak_ptr<CommPG> comm;
    static uint32_t comm_id;

    std::lock_guard<std::mutex> lock(m);
    id = (comm_id++ % TORCH_UCX_COMM_BITS);
    std::shared_ptr<CommPG> shared_comm = comm.lock();
    if (!shared_comm) {
      shared_comm = std::make_shared<CommPG>();
      comm = shared_comm;
    }
    return shared_comm;
  }

  void progress_loop() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop_progress_loop) {
      if (progress_list.empty()) {
        queue_produce_cv.wait(lock);
        continue;
      }
      auto work = progress_list.front();
      progress_list.pop_front();
      lock.unlock();
      queue_consume_cv.notify_one();

      do {
        ucp_worker_progress(ucx_comm.worker);
        ucc_context_progress(ucc_comm.context);
      } while (!work->isCompleted());
      lock.lock();
    }
  }
  torch_ucx_request_t* send_nb(
      ucp_ep_h ep,
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag);
  torch_ucx_request_t* recv_nb(
      void* data,
      ucs_memory_type_t mtype,
      size_t size,
      ucp_tag_t ucp_tag,
      ucp_tag_t ucp_tag_mask);
};

} // namespace c10d
