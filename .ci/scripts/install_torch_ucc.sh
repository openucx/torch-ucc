#!/bin/bash -eEx
set -o pipefail

echo "INFO: Install Torch-UCC"
cd "${TORCH_UCC_SRC_DIR}"
export UCX_HOME=${UCX_INSTALL_DIR}
export XCCL_HOME=${XCCL_INSTALL_DIR}
# TODO debug
export WITH_CUDA=${CUDA_HOME}
#export WITH_CUDA=no
python setup.py install bdist_wheel
pip3 list | grep torch
python -c 'import torch, torch_ucc'
cp "${TORCH_UCC_SRC_DIR}/dist/"*.whl "${TORCH_UCC_PKG_DIR}"
