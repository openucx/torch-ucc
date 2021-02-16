/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "torch_ucc.hpp"
#include <memory>

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

void check_tensor(const std::vector<at::Tensor>& tensors) {
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

ProcessGroupUCC::WorkUCX::~WorkUCX() {
  if (request_ != nullptr) {
    torch_ucx_request_free(request_);
  }
}

bool ProcessGroupUCC::WorkUCX::isCompleted() {
  if (request_ == nullptr) {
    return true;
  }
  return (request_->status == TORCH_UCX_REQUEST_DONE);
}

bool ProcessGroupUCC::WorkUCX::isSuccess() const {
  if (request_ == nullptr) {
    return true;
  }
  return (request_->status != TORCH_UCX_REQUEST_ERROR);
}

bool ProcessGroupUCC::WorkUCX::wait(std::chrono::milliseconds /* unused */) {
  while (!isCompleted())
    ;
  return true;
}

ProcessGroupUCC::WorkUCC::~WorkUCC() {
  if (request_ != nullptr) {
    ucc_collective_finalize(request_);
  }
}

bool ProcessGroupUCC::WorkUCC::isCompleted() {
  if (request_ == nullptr) {
    return true;
  }
  return (ucc_collective_test(request_) == UCC_OK);
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const {
  if (request_ == nullptr) {
    return true;
  }
  return (ucc_collective_test(request_) >= 0);
}

bool ProcessGroupUCC::WorkUCC::wait(std::chrono::milliseconds /* unused */) {
  while (!isCompleted())
    ;
  return true;
}

CommUCX::CommUCX() {
  ucp_params_t params;
  ucp_config_t* config;
  ucs_status_t st;
  ucp_worker_params_t worker_params;

  stop_progress_loop = false;
  st = ucp_config_read("TORCH", nullptr, &config);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to read UCP config: " << ucs_status_string(st);
    throw std::runtime_error(ucs_status_string(st));
  }
  memset(&params, 0, sizeof(ucp_params_t));
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
      UCP_PARAM_FIELD_ESTIMATED_NUM_EPS | UCP_PARAM_FIELD_TAG_SENDER_MASK |
      UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP;
  params.request_size = sizeof(torch_ucx_request_t);
  params.features = UCP_FEATURE_TAG;
  params.estimated_num_eps = 1; // TODO
  params.tag_sender_mask = TORCH_UCX_RANK_MASK;
  params.request_init = [](void* request) {
    static_cast<torch_ucx_request_t*>(request)->status =
        TORCH_UCX_REQUEST_ACTIVE;
  };
  params.request_cleanup = [](void*) {};
  st = ucp_init(&params, config, &context);
  ucp_config_release(config);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to init UCP context: " << ucs_status_string(st);
    throw std::runtime_error(ucs_status_string(st));
  }
  memset(&worker_params, 0, sizeof(ucp_worker_params_t));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(context, &worker_params, &worker);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to create UCP worker: " << ucs_status_string(st);
    ucp_cleanup(context);
    throw std::runtime_error(ucs_status_string(st));
  }
  progress_thread = std::thread(&CommUCX::progress_loop, this);
}

CommUCX::~CommUCX() {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(lock, [&] { return progress_list.empty(); });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
}

void CommUCX::connect_eps(
    std::vector<ucp_ep_h>& eps,
    int rank,
    int size,
    const c10::intrusive_ptr<Store>& store) {
  ucs_status_t st;
  ucp_address_t* local_addr;
  size_t local_addr_len;

  st = ucp_worker_get_address(worker, &local_addr, &local_addr_len);
  if (st != UCS_OK) {
    throw std::runtime_error(ucs_status_string(st));
  }
  auto key = "wa" + std::to_string(rank);
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  store->set(key, val);
  ucp_worker_release_address(worker, local_addr);
  eps.resize(size);
  for (int i = 0; i < size; i++) {
    std::vector<uint8_t> peer_addr = store->get("wa" + std::to_string(i));
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    st = ucp_ep_create(worker, &ep_params, &(eps[i]));
    if (st != UCS_OK) {
      throw std::runtime_error(ucs_status_string(st));
    }
  }
}

void CommUCX::disconnect_eps(
    std::vector<ucp_ep_h>& eps,
    const c10::intrusive_ptr<Store>& store) {
  ucs_status_t st;
  ucs_status_ptr_t close_req;

  for (ucp_ep_h& ep : eps) {
    close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(close_req)) {
      return;
    }
    if (UCS_PTR_IS_PTR(close_req)) {
      do {
        ucp_worker_progress(worker);
        st = ucp_request_check_status(close_req);
      } while (st != UCS_OK);
      ucp_request_free(close_req);
    }
  }
  auto key_ep_closed = "epclosed";
  auto num_closed_ep = store->add(key_ep_closed, 1);
  std::vector<std::string> key_finished{"finished"};
  if ((size_t)num_closed_ep == eps.size()) {
    store->add(key_finished[0], 1);
  } else {
    store->wait(key_finished);
  }
}

void CommUCX::progress_loop() {
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
      ucp_worker_progress(worker);
    } while (!work->isCompleted());
    lock.lock();
  }
}

torch_ucx_request_t* CommUCX::send_nb(
    ucp_ep_h ep,
    void* data,
    ucs_memory_type_t mtype,
    size_t size,
    ucp_tag_t ucp_tag) {
  ucs_status_ptr_t st;
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(size);
  params.memory_type = mtype;
  params.cb.send = [](void* request, ucs_status_t status, void* user_data) {
    static_cast<torch_ucx_request_t*>(request)->status = TORCH_UCX_REQUEST_DONE;
  };
  st = ucp_tag_send_nbx(ep, data, 1, ucp_tag, &params);
  if (torch_ucx_check_req(st) != TORCH_UCC_OK) {
    throw std::runtime_error("failed to send message");
  };
  return reinterpret_cast<torch_ucx_request_t*>(st);
}

torch_ucx_request_t* CommUCX::recv_nb(
    void* data,
    ucs_memory_type_t mtype,
    size_t size,
    ucp_tag_t ucp_tag,
    ucp_tag_t ucp_tag_mask) {
  ucs_status_ptr_t st;
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(size);
  params.cb.recv = [](void* request,
                      ucs_status_t status,
                      const ucp_tag_recv_info_t* info,
                      void* user_data) {
    static_cast<torch_ucx_request_t*>(request)->status = TORCH_UCX_REQUEST_DONE;
  };
  params.memory_type = mtype;
  st = ucp_tag_recv_nbx(worker, data, 1, ucp_tag, ucp_tag_mask, &params);
  if (torch_ucx_check_req(st) != TORCH_UCC_OK) {
    throw std::runtime_error("failed to recv message");
  };
  return reinterpret_cast<torch_ucx_request_t*>(st);
}

c10::intrusive_ptr<ProcessGroup::Work> CommUCX::enqueue_request(
    torch_ucx_request_t* request) {
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = progress_list.emplace(
      progress_list.end(),
      c10::make_intrusive<ProcessGroupUCC::WorkUCX>(request));
  lock.unlock();
  queue_produce_cv.notify_one();
  return (*iter);
}

CommUCC::CommUCC() {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  stop_progress_loop = false;
  st = ucc_lib_config_read("TORCH", nullptr, &lib_config);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to read UCC lib config: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_SINGLE;
  st = ucc_init(&lib_params, lib_config, &lib);
  ucc_lib_config_release(lib_config);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to init UCC lib: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    ucc_finalize(lib);
    LOG(ERROR) << "failed to read UCC context config: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE;
  context_params.ctx_type = UCC_CONTEXT_SHARED;
  ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  if (st != UCC_OK) {
    ucc_finalize(lib);
    LOG(ERROR) << "failed to create UCC context: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  progress_thread = std::thread(&CommUCC::progress_loop, this);
}

CommUCC::~CommUCC() {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(lock, [&] { return progress_list.empty(); });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
  ucc_context_destroy(context);
  ucc_finalize(lib);
}

void CommUCC::progress_loop() {
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
      ucc_context_progress(context);
    } while (!work->isCompleted());
    lock.lock();
  }
}

struct torch_ucc_oob_coll_info_t {
  const c10::intrusive_ptr<Store>* store;
  int rank;
  int size;
  void *rbuf;
  size_t msglen;
};

static ucc_status_t oob_allgather(void* sbuf, void* rbuf, size_t msglen,
                                  void* coll_info, void** req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(coll_info);
  LOG(ERROR) << "rank " << info->rank << ": starting allgather";
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(sbuf),
      reinterpret_cast<uint8_t*>(sbuf) + msglen);
  (*info->store)->set("teamr" + std::to_string(info->rank), val);
  info->rbuf = rbuf;
  info->msglen = msglen;
  *req = coll_info;
  return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);

  for (int r = 0; r < info->size; r++) {
    if (!((*info->store)->check({"teamr" + std::to_string(r)}))) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < info->size; r++) {
    std::vector<uint8_t> data = (*info->store)->get("teamr" + std::to_string(r));
    memcpy(
        (void*)((ptrdiff_t)info->rbuf + info->msglen * r),
        data.data(),
        info->msglen);
  }
  return UCC_OK;
}

static ucc_status_t oob_allgather_free(void *req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  LOG(ERROR) << "rank " << info->rank << ": removing key";
  uint64_t num_done = (*info->store)->add({"team_ag_done"}, 1);
  if (num_done == info->size) {
    (*info->store)->deleteKey("team_ag_done");
    for (int r = 0; r < info->size; r++) {
      if (r != info->rank){
        (*info->store)->add({"team_ag_finished" + std::to_string(r)}, 1);
      }
    }
  } else {
    (*info->store)->wait({"team_ag_finished" + std::to_string(info->rank)});
  }
  (*info->store)->deleteKey("teamr" + std::to_string(info->rank));
  (*info->store)->deleteKey("team_ag_finished" + std::to_string(info->rank));

  return UCC_OK;
}

void CommUCC::create_team(
      ucc_team_h &team,
      int rank,
      int size,
      const c10::intrusive_ptr<Store>& store) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  torch_ucc_oob_coll_info_t *coll_info = new torch_ucc_oob_coll_info_t;

  coll_info->rank = rank;
  coll_info->size = size;
  coll_info->store = &store;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
      UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = coll_info;
  team_params.oob.participants = size;
  team_params.ep = rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  st = ucc_team_create_post(&context, 1, &team_params, &team);
  if (st != UCC_OK) {
    delete coll_info;
    LOG(ERROR) << "failed to post team create: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  do {
      st = ucc_team_create_test(team);
  } while (st == UCC_INPROGRESS);
  if (st != UCC_OK) {
    delete coll_info;
    LOG(ERROR) << "failed to create UCC team: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  delete coll_info;
}

void CommUCC::destroy_team(
    ucc_team_h &team) {
  ucc_team_destroy(team);
}

c10::intrusive_ptr<ProcessGroup::Work> CommUCC::enqueue_request(
    ucc_coll_req_h request) {
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = progress_list.emplace(
      progress_list.end(),
      c10::make_intrusive<ProcessGroupUCC::WorkUCC>(request));
  lock.unlock();
  queue_produce_cv.notify_one();
  return (*iter);
}


std::shared_ptr<CommUCX> get_ucx_comm(uint32_t& id) {
  static std::mutex m;
  static std::weak_ptr<CommUCX> comm;
  static uint32_t last_tag;

  std::lock_guard<std::mutex> lock(m);
  id = (last_tag++ % TORCH_UCX_COMM_BITS);
  std::shared_ptr<CommUCX> shared_comm = comm.lock();
  if (!shared_comm) {
    shared_comm = std::make_shared<CommUCX>();
    comm = shared_comm;
  }
  return shared_comm;
}

std::shared_ptr<CommUCC> get_ucc_comm() {
  static std::mutex m;
  static std::weak_ptr<CommUCC> comm;

  std::lock_guard<std::mutex> lock(m);
  std::shared_ptr<CommUCC> shared_comm = comm.lock();
  if (!shared_comm) {
    shared_comm = std::make_shared<CommUCC>();
    comm = shared_comm;
  }
  return shared_comm;
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store) {
  ucx_comm_ = get_ucx_comm(ucx_tag);
  ucx_comm_->connect_eps(eps, rank, size, store);
  ucc_comm_ = get_ucc_comm();
  ucc_comm_->create_team(team, rank, size, store);
}

ProcessGroupUCC::~ProcessGroupUCC() {
  ucc_comm_->destroy_team(team);
  ucx_comm_->disconnect_eps(eps, store_);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support broadcast");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support allreduce");
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
  throw std::runtime_error("ProcessGroupUCC does not support allgather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& /* unused */) {
  ucc_coll_req_h request;
  ucc_coll_op_args_t coll;
  ucc_status_t st;

  LOG(ERROR) << "calling barrier";
  coll.coll_type = UCC_COLL_TYPE_BARRIER,
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

  return ucc_comm_->enqueue_request(request);
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
  throw std::runtime_error("ProcessGroupUCC does not support alltoall_base");
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
  ucp_tag_t ucp_tag;

  TORCH_UCX_MAKE_SEND_TAG(ucp_tag, tag, rank_, ucx_tag);
  torch_ucx_request_t* request = ucx_comm_->send_nb(
      eps[dstRank],
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag);
  return ucx_comm_->enqueue_request(request);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  ucp_tag_t ucp_tag, ucp_tag_mask;

  TORCH_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, srcRank, ucx_tag);
  torch_ucx_request_t* request = ucx_comm_->recv_nb(
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag,
      ucp_tag_mask);
  return ucx_comm_->enqueue_request(request);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  ucp_tag_t ucp_tag, ucp_tag_mask;

  TORCH_UCX_MAKE_RECV_TAG(
      ucp_tag, ucp_tag_mask, tag, TORCH_UCX_ANY_SOURCE, ucx_tag);
  torch_ucx_request_t* request = ucx_comm_->recv_nb(
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag,
      ucp_tag_mask);
  return ucx_comm_->enqueue_request(request);
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
