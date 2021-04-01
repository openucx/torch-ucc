# Torch-UCC
The Torch-UCC plugin is a research prototype that enables collective communication over [UCC](https://www.ucfconsortium.org/projects/ucc/) for distributed PyTorch applications that load the plugin at application runtime.

## Licenses
The torch-ucc plugin is licensed as:
* [BSD3](LICENSE)

## Contributor Agreement and Guidelines
In order to contribute to torch-ucc, please sign up with an appropriate
[Contributor Agreement](http://www.openucx.org/license/).
Follow these
[instructions](https://github.com/openucx/ucx/wiki/Guidance-for-contributors)
when submitting contributions and changes.


## Build
Required packages:
* PyTorch
* [UCX](https://github.com/openucx/ucx)
* [UCC](https://github.com/openucx/ucc)

```shell
# Build
UCX_HOME=<PATH_TO_UCX> UCC_HOME=<PATH_TO_UCC> WITH_CUDA=<PATH_TO_CUDA> python setup.py install
```
UCX_HOME required, specifies path to UCX installation directory

UCC_HOME required, specifies path to UCC installation directory

WITH_CUDA optional, if WITH_CUDA=no is set then only CPU tensors are supported

## Run
Configuration variables
| Name                               | Values                    | Description                                                                                                   |
|------------------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------|
| TORCH_UCC_ALLGATHER_BLOCKING_WAIT | 0 or 1                     | Sets behavior of wait function for CUDA Allgather. [Async collective in PyTorch](https://pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations)|
| TORCH_UCC_ALLREDUCE_BLOCKING_WAIT | 0 or 1                     | Sets behavior of wait function for CUDA Allreduce.                                                            |
| TORCH_UCC_ALLTOALL_BLOCKING_WAIT  | 0 or 1                     | Sets behavior of wait function for CUDA Alltoall.                                                             |
| TORCH_UCC_BCAST_BLOCKING_WAIT     | 0 or 1                     | Sets behavior of wait function for CUDA Bcast.                                                                |

```shell
export LD_LIBRARY_PATH=<PATH_TO_UCX>/lib:<PATH_TO_UCC>/lib:$LD_LIBRARY_PATH
python example.py
```

```python
import torch
import torch.distributed as dist
import torch_ucc

....
dist.init_process_group('ucc', rank=comm_rank, world_size=comm_size)
....
dist.all_to_all_single(recv_tensor, send_tensor)

```
