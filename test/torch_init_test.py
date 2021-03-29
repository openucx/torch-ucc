import os
import random
import torch
import torch.distributed as dist
import torch_ucc
import time
import sys

comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

os.environ['MASTER_PORT'] = '32167'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RANK']        = str(comm_rank)
os.environ['WORLD_SIZE']  = str(comm_size)
dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)
#dist.new_group(ranks=[0, 1], backend='ucc')
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
