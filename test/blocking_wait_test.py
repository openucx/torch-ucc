#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch_ucc

def init_pg(backend):
  global comm_rank, comm_size
  try:
    comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
  except:
    print('OMPI env variables are not found')
    sys.exit(1)
  torch.cuda.set_device(local_rank)

  os.environ['MASTER_PORT'] = '32167'
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['RANK']        = str(comm_rank)
  os.environ['WORLD_SIZE']  = str(comm_size)
  dist.init_process_group(backend, rank=comm_rank, world_size=comm_size)

def allreduce_test():
  global comm_rank, comm_size
  num_iters = 10
  dev = torch.device('cuda')
  t = torch.ones(100, device=dev)
  for i in range(10):
    dist.all_reduce(t)
  if torch.all(torch.eq(t, comm_size**num_iters)):
    print(f"Rank {comm_rank}: success")
  else:
    print(f"Rank {comm_rank}: failed")

def alltoall_test():
  global comm_rank, comm_size
  dev = torch.device('cuda')
  t_send = torch.zeros(comm_size, device=dev) + comm_rank
  t_recv = torch.zeros(comm_size, device=dev)
  dist.all_to_all_single(t_recv, t_send)
  t_recv = t_recv + 1
  dist.all_reduce(t_recv)
  if torch.all(torch.eq(t_recv, comm_size*torch.arange(start=1, end=comm_size+1, device=dev))):
    print(f"Rank {comm_rank}: success")
  else:
    print(f"Rank {comm_rank}: failed")


if __name__ == "__main__":
  if not torch.cuda.is_available():
    print("cuda is not available")
    sys.exit(1)
  parser = argparse.ArgumentParser(description="PG UCC nonblocking test")
  parser.add_argument("--backend", type=str, default='ucc')
  parser.add_argument("--test", type=str, default='ucc')
  args = parser.parse_args()

  comm_rank = -1
  comm_size = -1
  init_pg(args.backend)
  if args.test == "allreduce":
    allreduce_test()
  elif args.test == "alltoall":
    alltoall_test()
  else:
    print("Wrong test name")
