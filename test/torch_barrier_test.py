#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import time
import sys
import random
from torch_ucc_test_setup import *

args = parse_test_args()
pg = init_process_groups(args.backend, args.use_cuda)

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

for i in range(comm_size):
    rand_sleep = random.randint(1, 1000)
    time.sleep(rand_sleep/1000)
    if i == comm_rank:
        print("rank {} checks in".format(comm_rank))
        sys.stdout.flush()
    dist.barrier()
dist.barrier()
if comm_rank == 0:
    print("Test barrier: succeeded")
