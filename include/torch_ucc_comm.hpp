/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

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

namespace c10d {

struct torch_ucc_oob_coll_info_t {
  c10::intrusive_ptr<Store> store;
  int rank;
  int size;
  void* rbuf;
  size_t msglen;
};

class CommBase {
 public:
  CommBase() {}
  virtual void progress() = 0;
  virtual ~CommBase() {}
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
  CommUCC(torch_ucc_oob_coll_info_t* oob_info);
  ~CommUCC();
};

ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req);

ucc_status_t oob_allgather_test(void* req);

ucc_status_t oob_allgather_free(void* req);


} // namespace c10d
