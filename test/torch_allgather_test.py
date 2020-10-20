#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import sys
import numpy as np
from torch_ucc_test_setup import *

args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = 2 ** np.arange(24)
print_test_head("Allgather", comm_rank)
for count in counts:
    tensor_input = get_tensor(count, args.use_cuda)
    tensors_out_ucc = []
    tensors_out_test = []
    for p in range(comm_size):
        tensors_out_ucc.append(get_tensor(count, args.use_cuda));
        tensors_out_test.append(get_tensor(count, args.use_cuda));
    dist.all_gather(tensors_out_ucc, tensor_input)
    dist.all_gather(tensors_out_test, tensor_input, group=pg)
    status = check_tensor_list_equal(tensors_out_ucc, tensors_out_test)
    dist.all_reduce(status, group=pg)
    print_test_result(status, count, comm_rank, comm_size)
if comm_rank == 0:
    print("Test allgather: succeeded")
