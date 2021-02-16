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
#include <ucp/api/ucp.h>
#include <ucc/api/ucc.h>
#include "torch_ucc_sendrecv.hpp"

namespace c10d {

class CommUCX;

class CommUCC;

class ProcessGroupUCC : public ProcessGroup {
 public:
  class WorkUCX : public ProcessGroup::Work {
    friend class ProcessGroupUCC;
    friend class CommUCX;

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
  std::shared_ptr<CommUCX> ucx_comm_;
  std::shared_ptr<CommUCC> ucc_comm_;
  uint32_t ucx_tag;
  std::vector<ucp_ep_h> eps;
  ucc_team_h team;
};

class CommUCX {
 protected:
  ucp_context_h context;
  ucp_worker_h worker;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::list<c10::intrusive_ptr<ProcessGroupUCC::WorkUCX>> progress_list;
  bool stop_progress_loop;
  void progress_loop();

 public:
  CommUCX();
  ~CommUCX();
  void connect_eps(
      std::vector<ucp_ep_h>& eps,
      int rank,
      int size,
      const c10::intrusive_ptr<Store>& store);
  void disconnect_eps(
      std::vector<ucp_ep_h>& eps,
      const c10::intrusive_ptr<Store>& store);
  c10::intrusive_ptr<ProcessGroup::Work> enqueue_request(
      torch_ucx_request_t* request);
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
 protected:
  ucc_lib_h lib;
  ucc_context_h context;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::list<c10::intrusive_ptr<ProcessGroupUCC::WorkUCC>> progress_list;
  bool stop_progress_loop;
  void progress_loop();

 public:
  CommUCC();
  ~CommUCC();
  void create_team(
      ucc_team_h &team,
      int rank,
      int size,
      const c10::intrusive_ptr<Store>& store);
  void destroy_team(
      ucc_team_h &team);
  c10::intrusive_ptr<ProcessGroup::Work> enqueue_request(
      ucc_coll_req_h request);
};

class CommPG {
  CommUCX ucx_comm;
  CommUCC ucc_comm;

}


} // namespace c10d
