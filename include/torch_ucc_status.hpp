/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

namespace c10d {

enum torch_ucc_status_t {
  TORCH_UCC_OK = 0,
  TORCH_UCC_INPROGRESS = 1,
  TORCH_UCC_OPERATION_INITIALIZED = 2,
  TORCH_UCC_ERROR = -1,
};

};
