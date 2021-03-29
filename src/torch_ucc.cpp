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

const std::map<c10::DeviceType, ucc_memory_type_t> ucc_mtype_map = {
    {c10::kCPU, UCC_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCC_MEMORY_TYPE_CUDA},
    {c10::kHIP, UCC_MEMORY_TYPE_ROCM},
    {c10::kFPGA, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kMSNPU, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kXLA, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kVulkan, UCC_MEMORY_TYPE_UNKNOWN},
    {c10::kMetal, UCC_MEMORY_TYPE_UNKNOWN},
};

const std::map<at::ScalarType, ucc_datatype_t> ucc_dtype_map = {
    {at::kByte, UCC_DT_UINT8},
    {at::kChar, UCC_DT_INT8},
    {at::kHalf, UCC_DT_FLOAT16},
    {at::kDouble, UCC_DT_FLOAT64},
    {at::kFloat, UCC_DT_FLOAT32},
    {at::kInt, UCC_DT_INT32},
    {at::kLong, UCC_DT_INT64},
};

const std::map<ReduceOp, ucc_reduction_op_t> ucc_op_map = {
    {ReduceOp::SUM, UCC_OP_SUM},
    {ReduceOp::PRODUCT, UCC_OP_PROD},
    {ReduceOp::MIN, UCC_OP_MIN},
    {ReduceOp::MAX, UCC_OP_MAX},
    {ReduceOp::BAND, UCC_OP_BAND},
    {ReduceOp::BOR, UCC_OP_BOR},
    {ReduceOp::BXOR, UCC_OP_BXOR},
};

void check_device(c10::Device dev1, c10::Device dev2) {
  if (dev1.is_cuda() && dev2.is_cuda() && dev1 != dev2) {
    throw std::runtime_error("ProcessGroupUCC multidevice is not supported");
  }
}

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

ProcessGroupUCC::WorkUCC::~WorkUCC() {
  TORCH_CHECK(request_ == nullptr, "TorchUCC, request wasn't finalized");
}

bool ProcessGroupUCC::WorkUCC::isCompleted() {
  return (status_ != UCC_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const {
  return (status_ >= 0);
}

bool ProcessGroupUCC::WorkUCC::wait(std::chrono::milliseconds /* unused */) {
  while (!isCompleted())
    ;
  return true;
}

void ProcessGroupUCC::WorkUCC::finalize() {
  if (request_ != nullptr) {
    if (isP2POp(opType_)) {
      request_->status = UCC_INPROGRESS;
      ucp_request_free(request_);
    } else {
      ucc_collective_finalize(request_);
    }
    status_ = UCC_OK;
    request_ = nullptr;
  }
}

CommUCX::CommUCX(int comm_size) {
  ucp_params_t params;
  ucp_config_t* config;
  ucs_status_t st;
  ucp_worker_params_t worker_params;

  st = ucp_config_read("TORCH", nullptr, &config);
  if (st != UCS_OK) {
    LOG(ERROR) << "failed to read UCP config: " << ucs_status_string(st);
    throw std::runtime_error(ucs_status_string(st));
  }
  memset(&params, 0, sizeof(ucp_params_t));
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
      UCP_PARAM_FIELD_ESTIMATED_NUM_EPS | UCP_PARAM_FIELD_TAG_SENDER_MASK |
      UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP;
  params.request_size = sizeof(ucc_coll_req_t);
  params.features = UCP_FEATURE_TAG;
  params.estimated_num_eps = comm_size;
  params.tag_sender_mask = TORCH_UCX_RANK_MASK;
  params.request_init = [](void* request) {
    static_cast<ucc_coll_req_h>(request)->status = UCC_INPROGRESS;
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
}

void CommUCX::progress() {
  ucp_worker_progress(worker);
}

CommUCX::~CommUCX() {
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
}

void CommPG::ucx_connect_eps(
    std::vector<ucp_ep_h>& eps,
    int rank,
    int size,
    const c10::intrusive_ptr<Store>& store) {
  ucs_status_t st;
  ucp_address_t* local_addr;
  size_t local_addr_len;

  st = ucp_worker_get_address(ucx_comm.worker, &local_addr, &local_addr_len);
  if (st != UCS_OK) {
    throw std::runtime_error(ucs_status_string(st));
  }
  auto key = "wa" + std::to_string(rank);
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  store->set(key, val);
  ucp_worker_release_address(ucx_comm.worker, local_addr);
  eps.resize(size);
  for (int i = 0; i < size; i++) {
    std::vector<uint8_t> peer_addr = store->get("wa" + std::to_string(i));
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    st = ucp_ep_create(ucx_comm.worker, &ep_params, &(eps[i]));
    if (st != UCS_OK) {
      throw std::runtime_error(ucs_status_string(st));
    }
  }
}

void CommPG::ucx_disconnect_eps(
    std::vector<ucp_ep_h>& eps,
    const c10::intrusive_ptr<Store>& store) {
  ucs_status_t st;

  for (ucp_ep_h& ep : eps) {
    ucs_status_ptr_t close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(close_req)) {
      LOG(ERROR) << "failed to close endpoint";
      return;
    }
    if (UCS_PTR_IS_PTR(close_req)) {
      do {
        ucp_worker_progress(ucx_comm.worker);
        st = ucp_request_check_status(close_req);
      } while (st != UCS_OK);
      ucp_request_free(close_req);
    }
  }
  if ((size_t)store->add("epclosed", 1) == eps.size()) {
    store->add("finished", 1);
  } else {
    store->wait({"finished"});
  }
}

ucc_coll_req_h CommPG::send_nb(
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
    static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
  };
  st = ucp_tag_send_nbx(ep, data, 1, ucp_tag, &params);
  if (UCS_PTR_IS_ERR(st)) {
    LOG(ERROR) << "failed to send message: "
               << ucs_status_string(UCS_PTR_STATUS(st));
    throw std::runtime_error(ucs_status_string(UCS_PTR_STATUS(st)));
  }
  return reinterpret_cast<ucc_coll_req_h>(st);
}

ucc_coll_req_h CommPG::recv_nb(
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
    static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
  };
  params.memory_type = mtype;
  st = ucp_tag_recv_nbx(
      ucx_comm.worker, data, 1, ucp_tag, ucp_tag_mask, &params);
  if (UCS_PTR_IS_ERR(st)) {
    LOG(ERROR) << "failed to recv message: "
               << ucs_status_string(UCS_PTR_STATUS(st));
    throw std::runtime_error(ucs_status_string(UCS_PTR_STATUS(st)));
  }
  return reinterpret_cast<ucc_coll_req_h>(st);
}

CommUCC::CommUCC(int comm_size) {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

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
    LOG(ERROR) << "failed to read UCC context config: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_context_config_modify(context_config, NULL, "ESTIMATED_NUM_EPS",
                                 std::to_string(comm_size).c_str());
  if (st != UCC_OK) {
    ucc_context_config_release(context_config);
    ucc_finalize(lib);
    LOG(ERROR) << "failed to modify UCC context config: "
               << ucc_status_string(st);
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
}

void CommUCC::progress() {
  ucc_context_progress(context);
}

CommUCC::~CommUCC() {
  ucc_context_destroy(context);
  ucc_finalize(lib);
}

struct torch_ucc_oob_coll_info_t {
  const c10::intrusive_ptr<Store>* store;
  int rank;
  int size;
  void* rbuf;
  size_t msglen;
};

static ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(coll_info);
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(sbuf),
      reinterpret_cast<uint8_t*>(sbuf) + msglen);
  (*info->store)->set("teamr" + std::to_string(info->rank), val);
  info->rbuf = rbuf;
  info->msglen = msglen;
  *req = coll_info;
  return UCC_OK;
}

static ucc_status_t oob_allgather_test(void* req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);

  for (int r = 0; r < info->size; r++) {
    if (!((*info->store)->check({"teamr" + std::to_string(r)}))) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < info->size; r++) {
    std::vector<uint8_t> data =
        (*info->store)->get("teamr" + std::to_string(r));
    memcpy(
        (void*)((ptrdiff_t)info->rbuf + info->msglen * r),
        data.data(),
        info->msglen);
  }
  return UCC_OK;
}

static ucc_status_t oob_allgather_free(void* req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  int num_done = (*info->store)->add({"team_ag_done"}, 1);
  if (num_done == info->size) {
    (*info->store)->deleteKey("team_ag_done");
    for (int r = 0; r < info->size; r++) {
      if (r != info->rank) {
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

void CommPG::ucc_create_team(
    ucc_team_h& team,
    int rank,
    int size,
    const c10::intrusive_ptr<Store>& store) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  torch_ucc_oob_coll_info_t* coll_info = new torch_ucc_oob_coll_info_t;

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
  st = ucc_team_create_post(&ucc_comm.context, 1, &team_params, &team);
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
  // TODO: don't delete
  delete coll_info;
}

void CommPG::ucc_destroy_team(ucc_team_h& team) {
  ucc_team_destroy(team);
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store) {
  comm = nullptr;
}

ProcessGroupUCC::~ProcessGroupUCC() {
  comm->ucc_destroy_team(team);
  comm->ucx_disconnect_eps(eps, store_);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  auto& tensor = inputTensors[0];
  check_device(tensor.device(), outputTensors[0][0].device());
  initComm(tensor.device());

  AllgatherWorkData* data = new AllgatherWorkData(size_);
  for (int i = 0; i < size_; i++) {
    data->recv_lengths[i] = tensor.element_size() * tensor.numel();
    data->recv_offsets[i] = (uint64_t)outputTensors[0][i].data_ptr();
  }
  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_ALLGATHERV;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.element_size() * tensor.numel();
  coll.src.info.datatype = UCC_DT_UINT8;
  coll.src.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  coll.dst.info_v.buffer = nullptr;
  coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
  coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
  coll.dst.info_v.datatype = UCC_DT_UINT8;
  coll.dst.info_v.mem_type =
      ucc_mtype_map.at(outputTensors[0][0].device().type());
  return comm->enqueue_collective(
      OpType::ALLGATHER, coll, std::unique_ptr<WorkData>(data), team);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS |
              UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_ALLREDUCE;
  coll.reduce.predefined_op = ucc_op_map.at(opts.reduceOp);
  coll.src.info.buffer = nullptr;
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.src.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  coll.dst.info.buffer = tensor.data_ptr();
  coll.dst.info.count = tensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.dst.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  return comm->enqueue_collective(OpType::ALLREDUCE, coll, nullptr, team);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_device(inputTensor.device(), outputTensor.device());
  initComm(inputTensor.device());
  ucc_coll_args_t coll;
  AlltoallWorkData* data;

  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    data = nullptr;
    TORCH_CHECK(
        (outputTensor.size(0) % size_ == 0) &&
            (inputTensor.size(0) % size_ == 0),
        "Tensor's dim 0 does not divide equally across group size");
    coll.mask = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALL;
    coll.src.info.buffer = inputTensor.data_ptr();
    coll.src.info.count =
        inputTensor.element_size() * inputTensor.numel() / size_;
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = ucc_mtype_map.at(inputTensor.device().type());
    coll.dst.info.buffer = outputTensor.data_ptr();
    coll.dst.info.count =
        outputTensor.element_size() * outputTensor.numel() / size_;
    coll.dst.info.datatype = UCC_DT_UINT8;
    coll.dst.info.mem_type = ucc_mtype_map.at(outputTensor.device().type());
  } else {
    AlltoallWorkData* data = new AlltoallWorkData(size_);
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    computeLengthsAndOffsets(
        outputSplitSizes,
        outputTensor,
        &data->recv_lengths,
        &data->recv_offsets);
    computeLengthsAndOffsets(
        inputSplitSizes, inputTensor, &data->send_lengths, &data->send_offsets);
    coll.mask = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
    coll.src.info_v.buffer = inputTensor.data_ptr();
    coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
    coll.src.info_v.datatype = ucc_dtype_map.at(inputTensor.scalar_type());
    coll.src.info_v.mem_type = ucc_mtype_map.at(inputTensor.device().type());
    coll.dst.info_v.buffer = outputTensor.data_ptr();
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = ucc_dtype_map.at(outputTensor.scalar_type());
    coll.dst.info_v.mem_type = ucc_mtype_map.at(outputTensor.device().type());
  }
  return comm->enqueue_collective(
      OpType::ALLTOALL_BASE, coll, std::unique_ptr<WorkData>(data), team);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& /* unused */) {
  initComm(c10::DeviceType::CPU);

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.coll_type = UCC_COLL_TYPE_BARRIER;
  return comm->enqueue_collective(OpType::BARRIER, coll, nullptr, team);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type());
  coll.src.info.mem_type = ucc_mtype_map.at(tensor.device().type());
  coll.root = opts.rootRank;
  return comm->enqueue_collective(OpType::BROADCAST, coll, nullptr, team);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucp_tag_t ucp_tag;
  TORCH_UCX_MAKE_SEND_TAG(ucp_tag, tag, rank_, comm_id);
  ucc_coll_req_h request = comm->send_nb(
      eps[dstRank],
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag);
  return comm->enqueue_p2p(OpType::SEND, request);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucp_tag_t ucp_tag, ucp_tag_mask;
  TORCH_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, srcRank, comm_id);
  ucc_coll_req_h request = comm->recv_nb(
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag,
      ucp_tag_mask);
  return comm->enqueue_p2p(OpType::RECV, request);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  check_tensor(tensors);
  auto& tensor = tensors[0];
  initComm(tensor.device());

  ucp_tag_t ucp_tag, ucp_tag_mask;
  TORCH_UCX_MAKE_RECV_TAG(
      ucp_tag, ucp_tag_mask, tag, TORCH_UCX_ANY_SOURCE, comm_id);
  ucc_coll_req_h request = comm->recv_nb(
      tensor.data_ptr(),
      ucs_mtype_map.at(tensor.device().type()),
      tensor.numel() * tensor.element_size(),
      ucp_tag,
      ucp_tag_mask);
  return comm->enqueue_p2p(OpType::RECVANYSOURCE, request);
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size);
}

void ProcessGroupUCC::initComm(c10::Device dev) {
  if (!comm) {
    comm = CommPG::get_comm(comm_id, dev, size_);
    comm->ucx_connect_eps(eps, rank_, size_, store_);
    comm->ucc_create_team(team, rank_, size_, store_);
  } else {
    if (dev.is_cuda()) {
      if ((comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
          (comm->cuda_device_index != dev.index())) {
        LOG(ERROR)
            << "ucc communicator was initialized with different cuda device,"
            << "multi device is not supported";
        throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
      }
      comm->cuda_device_index = dev.index();
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &ProcessGroupUCC::createProcessGroupUCC);
}

} // namespace c10d
