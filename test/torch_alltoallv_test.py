#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

from torch_ucc_test_setup import *
import torch.autograd.profiler as profiler
import numpy as np

args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = 2 ** np.arange(4, 24)

print_test_head("Alltoallv", comm_rank)
for count in counts:
    np.random.seed(3131)

    split = np.random.randint(low=1, high=2*count//comm_size, size=(comm_size,comm_size))
    input_size = np.sum(split, axis=1)
    output_size = np.sum(split, axis=0)

    send_tensor = get_tensor(input_size[comm_rank], args.use_cuda)
    recv_tensor = get_tensor(output_size[comm_rank], args.use_cuda)
    recv_tensor_test = get_tensor(output_size[comm_rank], args.use_cuda)
    dist.all_to_all_single(recv_tensor, send_tensor,
                          split[:, comm_rank],
                          split[comm_rank, :])
    dist.all_to_all_single(recv_tensor_test, send_tensor,
                          split[:, comm_rank],
                          split[comm_rank, :],
                          group=pg)
    status = check_tensor_equal(recv_tensor, recv_tensor_test)
    dist.all_reduce(status, group=pg)
    print_test_result(status, "{}({})".format(count, input_size[comm_rank]), comm_rank, comm_size)
if comm_rank == 0:
    print("Test alltoallv: succeeded")
