#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

from torch_ucc_test_setup import *
import numpy as np

args = parse_test_args()
pg = init_process_groups(args.backend)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

counts = 2 ** np.arange(4, 24)

for count in counts:
    np.random.seed(3131)

    split = np.random.randint(low=1, high=2*count//comm_size, size=(comm_size,comm_size))
    input_size = np.sum(split, axis=1)
    output_size = np.sum(split, axis=0)

    send_tensor = get_tensor(input_size[comm_rank])
    recv_tensor = get_tensor(output_size[comm_rank])
    recv_tensor_test = get_tensor(output_size[comm_rank])
    dist.all_to_all_single(recv_tensor, send_tensor, 
                           split[:, comm_rank],
                           split[comm_rank, :])
    dist.all_to_all_single(recv_tensor_test, send_tensor, 
                           split[:, comm_rank],
                           split[comm_rank, :],
                           group=pg)
    check_tensor_equal("alltoallv", recv_tensor, recv_tensor_test)

if comm_rank == 0:
    print("Test alltoallv: succeeded")
