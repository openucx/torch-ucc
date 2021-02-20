import os
import torch
import torch.distributed as dist
import torch_ucc

comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

if comm_size != 2:
  print("sendrecv rest requires exactly 2 ranks")
  sys.exit(0)

os.environ['MASTER_PORT'] = '32167'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RANK']        = str(comm_rank)
os.environ['WORLD_SIZE']  = str(comm_size)
dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)

if comm_rank == 0:
  t = torch.full([16], comm_rank + 1)
  print("send: ", t)
  dist.send(t, 1, tag=128)
if comm_rank == 1:
  t = torch.full([16], 0)
  print("recv before: ", t)
  dist.recv(t, 0, tag=128)
  print("recv after: ", t)