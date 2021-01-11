# Torch-UCC
The Torch-UCC plugin is a research prototype that enables collective communication over XCCL for distributed PyTorch applications that load the plugin at application runtime. The XCCL interface is a non-standard API with a corresponding [reference implementation](https://github.com/openucx/xccl) used to guide design decisions for the [UCC project](https://www.ucfconsortium.org/projects/ucc/).

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
* [XCCL](https://github.com/openucx/xccl)

```shell
# Build
UCX_HOME=<PATH_TO_UCX> XCCL_HOME=<PATH_TO_XCCL> WITH_CUDA=<PATH_TO_CUDA> python setup.py clean --all install
```
UCX_HOME required, specifies path to UCX installation directory

XCCL_HOME required, specifies path to XCCL installation directory

WITH_CUDA optional, if WITH_CUDA=no is set then only CPU tensors are supported

## Run
Configuration variables
| Name                           | Values                      | Description                                                                                                   |
|--------------------------------|-----------------------------|---------------------------------------------------------------------------------------------------------------|
| TORCH_UCC_THREAD_ENABLE        | 0 or 1                      | If not equal to zero then dedicated thread will be used to progress point to point and collective operations. |
| TORCH_UCC_TLS                  | list of xccl team libraries | Allows to choose what xccl team libraries will be used |
| TORCH_UCC_BLOCKING_WAIT        | 0 or 1                      | Experimenal, defines behavior of wait function for cuda buffers |
| TORCH_UCC_HIGH_PRIORITY_STREAM | 0 or 1                      | Internal stream priority, relevant only when BLOCKING_WAIT is set to zero |

```shell
export LD_LIBRARY_PATH=<PATH_TO_UCX>/lib:<PATH_TO_XCCL>/lib:$LD_LIBRARY_PATH
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
