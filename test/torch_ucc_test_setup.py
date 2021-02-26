#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

import argparse
import torch
import torch.distributed as dist
import torch_ucc
import sys
import os

def parse_test_args():
    parser = argparse.ArgumentParser(description="PG UCC Test")
    parser.add_argument("--backend", type=str, default='mpi')
    parser.add_argument("--use-cuda", default=False, action='store_true')
    parser.add_argument("--enable-prof",default=False, action='store_true')
    args = parser.parse_args()

    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(0)

    return args


def get_tensor(count, is_cuda):
    dev = torch.device('cuda') if is_cuda else torch.device('cpu')
    t = torch.randint(0, 100, (count,), dtype=torch.int, device=dev)
    return t

def init_process_groups(bend, use_cuda):
    try:
        comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    except:
        print('OMPI env variables are not found')
        sys.exit(1)

    if use_cuda:
        torch.cuda.set_device(local_rank)

    os.environ['MASTER_PORT'] = '32167'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['RANK']        = str(comm_rank)
    os.environ['WORLD_SIZE']  = str(comm_size)
    dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)
    pg = dist.new_group(backend=bend)

    return pg

def check_tensor_equal(t1, t2):
    if torch.all(torch.eq(t1, t2)):
        return torch.tensor(1, device=t1.device)
    else:
        return torch.tensor(0, device=t1.device)

def check_tensor_list_equal(t1, t2):
    num_tensors = len(t1)
    for i in range(num_tensors):
        if not torch.all(torch.eq(t1[i], t2[i])):
            return torch.tensor(0, device=t1[i].device)
    return torch.tensor(1, device=t1[i].device)

def print_test_head(test_name, comm_rank):
    if comm_rank == 0:
        print("{} test".format(test_name))
        print("{0:20} {1}".format("count", "result"))

def print_test_result(status, count, comm_rank, comm_size):
    if comm_rank == 0:
        result = "OK" if status == comm_size else "Failed"
        print("{0:20} {1}".format(str(count), result))
    if status != comm_size:
        sys.exit(1)
