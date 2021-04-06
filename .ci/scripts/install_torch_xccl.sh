#!/bin/bash -eEx
set -o pipefail

# XCCL
echo "INFO: Install Torch-UCC (XCCL version)"
cd "${TORCH_UCC_PYTHON_VENV_DIR}"
python3 -m venv --system-site-packages xccl
. "${TORCH_UCC_PYTHON_VENV_DIR}/xccl/bin/activate"
export UCX_HOME=${UCX_INSTALL_DIR}
export XCCL_HOME=${XCCL_INSTALL_DIR}
export WITH_CUDA=${CUDA_HOME}
TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT="${TORCH_UCC_SRC_DIR}_xccl"
mkdir -p "${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}"
git clone https://github.com/openucx/torch-ucc.git "${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}"
cd "${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}"
git clean -ffdx
python setup.py install bdist_wheel
pip3 list | grep torch
python -c 'import torch, torch_ucc'
cp "${TORCH_UCC_SRC_DIR_WITH_XCCL_SUPPORT}/dist/"*.whl "${TORCH_UCC_PKG_DIR}"
deactivate
