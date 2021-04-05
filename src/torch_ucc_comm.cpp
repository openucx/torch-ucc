/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "torch_ucc_comm.hpp"

namespace c10d {

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

ucc_status_t oob_allgather(
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
  info->store->set("teamr" + std::to_string(info->rank), val);
  info->rbuf = rbuf;
  info->msglen = msglen;
  *req = coll_info;
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void* req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);

  for (int r = 0; r < info->size; r++) {
    if (!(info->store->check({"teamr" + std::to_string(r)}))) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < info->size; r++) {
    std::vector<uint8_t> data =
        info->store->get("teamr" + std::to_string(r));
    memcpy(
        (void*)((ptrdiff_t)info->rbuf + info->msglen * r),
        data.data(),
        info->msglen);
  }
  return UCC_OK;
}

ucc_status_t oob_allgather_free(void* req) {
  torch_ucc_oob_coll_info_t* info =
      reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  int num_done = info->store->add({"team_ag_done"}, 1);
  if (num_done == info->size) {
    info->store->deleteKey("team_ag_done");
    for (int r = 0; r < info->size; r++) {
      info->store->deleteKey("teamr" + std::to_string(r));
    }
    for (int r = 0; r < info->size; r++) {
      info->store->add({"team_ag_finished" + std::to_string(r)}, 1);
    }
  } else {
    info->store->wait({"team_ag_finished" + std::to_string(info->rank)});
  }
  info->store->deleteKey("team_ag_finished" + std::to_string(info->rank));

  return UCC_OK;
}

CommUCC::CommUCC(torch_ucc_oob_coll_info_t* oob_info) {
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
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  st = ucc_init(&lib_params, lib_config, &lib);
  ucc_lib_config_release(lib_config);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to init UCC lib: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  st = ucc_lib_get_attr(lib, &lib_attr);
  if (st != UCC_OK) {
    LOG(ERROR) << "failed to query for lib attr: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  if (lib_attr.thread_mode != UCC_THREAD_MULTIPLE) {
    LOG(ERROR) << "ucc library wasn't initialized with mt support "
               << "check ucc compile options ";
    throw std::runtime_error("failed to init ucc lib");
  }
  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    ucc_finalize(lib);
    LOG(ERROR) << "failed to read UCC context config: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_context_config_modify(
      context_config,
      NULL,
      "ESTIMATED_NUM_EPS",
      std::to_string(oob_info->size).c_str());
  if (st != UCC_OK) {
    ucc_context_config_release(context_config);
    ucc_finalize(lib);
    LOG(ERROR) << "failed to modify UCC context config: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.participants = oob_info->size;
  context_params.oob.allgather = oob_allgather;
  context_params.oob.req_test = oob_allgather_test;
  context_params.oob.req_free = oob_allgather_free;
  context_params.oob.coll_info = oob_info;
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

} // namespace c10d
