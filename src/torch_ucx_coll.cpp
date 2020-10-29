/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include <torch_ucx_coll.hpp>
#include <torch_ucc_ops.hpp>
#include <cstdlib>

namespace c10d {

static void torch_ucx_get_coll_config(torch_ucx_coll_config_t* config) {
  char* env;

  config->chunk = 1;
  config->reverse = 0;
  config->max_polls = 10;

  env = std::getenv("TORCH_UCC_UCX_CHUNK");
  if (env) {
    config->chunk = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_UCX_REVERSE");
  if (env) {
    config->reverse = std::atoi(env);
  }
  env = std::getenv("TORCH_UCC_UCX_MAX_POLLS");
  if (env) {
    config->max_polls = std::atoi(env);
  }
}

torch_ucc_status_t torch_ucx_coll_comm_init(
    torch_ucx_comm_t* p2p_comm,
    torch_ucc_coll_comm_t** comm) {
  torch_ucx_coll_comm_t* coll_comm;

  coll_comm = new torch_ucx_coll_comm_t;
  torch_ucx_get_coll_config(&coll_comm->config);
  coll_comm->p2p_comm = p2p_comm;
  coll_comm->last_tag = 0;
  *comm = (torch_ucc_coll_comm_t*)coll_comm;
  return TORCH_UCC_OK;
}

torch_ucc_status_t torch_ucx_coll_test(torch_ucc_coll_request_t* request) {
  torch_ucx_coll_request_t* req = (torch_ucx_coll_request_t*)request;

  return req->status;
}

torch_ucc_status_t torch_ucx_coll_comm_close(torch_ucc_coll_comm_t* comm) {
  torch_ucx_coll_comm_t* coll_comm;

  coll_comm = (torch_ucx_coll_comm_t*)comm;
  delete coll_comm;

  return TORCH_UCC_OK;
}

torch_ucc_status_t torch_ucx_coll_progress(torch_ucc_coll_request_t* request) {
  torch_ucx_coll_request_t* req = (torch_ucx_coll_request_t*)request;

  return req->progress(req);
}

torch_ucc_status_t torch_ucx_coll_free(torch_ucc_coll_request_t* request) {
  torch_ucx_coll_request_t* req = (torch_ucx_coll_request_t*)request;

  delete req;
  return TORCH_UCC_OK;
}

torch_ucc_status_t torch_ucx_coll_allreduce(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& tensor,
    const AllreduceOptions& opts,
    torch_ucc_coll_request_t** request) {
  fprintf(stderr, "ProcessGroupUCC: UCX backend doesn't support allreduce\n");
  return TORCH_UCC_ERROR;
}

torch_ucc_status_t torch_ucx_coll_barrier(
    torch_ucc_coll_comm_t* coll_comm,
    torch_ucc_coll_request_t** request) {
  fprintf(stderr, "ProcessGroupUCC: UCX backend doesn't support barrier\n");
  return TORCH_UCC_ERROR;
}

torch_ucc_status_t torch_ucx_coll_allgather(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& input_tensor,
    std::vector<at::Tensor>& output_tensors,
    torch_ucc_coll_request_t** request) {
  fprintf(stderr, "ProcessGroupUCC: UCX backend doesn't support allgather\n");
  return TORCH_UCC_ERROR;
}

torch_ucc_status_t torch_ucx_coll_broadcast(
    torch_ucc_coll_comm_t* coll_comm,
    at::Tensor& tensor,
    int root,
    torch_ucc_coll_request_t** request) {
  fprintf(stderr, "ProcessGroupUCC: UCX backend doesn't support broadcast\n");
  return TORCH_UCC_ERROR;
}

torch_ucc_coll_ops_t ucx_coll_ops{torch_ucx_coll_comm_init,
                                  torch_ucx_coll_allgather,
                                  torch_ucx_alltoall,
                                  torch_ucx_alltoallv,
                                  torch_ucx_coll_allreduce,
                                  torch_ucx_coll_barrier,
                                  torch_ucx_coll_broadcast,
                                  torch_ucx_coll_progress,
                                  torch_ucx_coll_test,
                                  torch_ucx_coll_free,
                                  torch_ucx_coll_comm_close};

} // namespace c10d
