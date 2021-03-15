#!/bin/bash -eEx
set -o pipefail

echo "INFO: Build XCCL"
cd "${TORCH_UCC_SRC_DIR}/xccl"
"${TORCH_UCC_SRC_DIR}/xccl/autogen.sh"
mkdir -p "${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}"
cd "${TORCH_UCC_SRC_DIR}/xccl/build-${XCCL_BUILD_TYPE}"
# TODO enable CUDA (compilation failed)
#"${TORCH_UCC_SRC_DIR}/xccl/configure" --with-ucx="${UCX_INSTALL_DIR}" --prefix="${XCCL_INSTALL_DIR}" --enable-debug
"${TORCH_UCC_SRC_DIR}/xccl/configure" --with-cuda="${CUDA_HOME}" --with-ucx="${UCX_INSTALL_DIR}" \
    --prefix="${XCCL_INSTALL_DIR}" --enable-debug
make -j install
echo "${XCCL_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/xccl.conf
ldconfig
ldconfig -p | grep -i xccl
make -C test
cd "${XCCL_INSTALL_DIR}" && tar cfz "${TORCH_UCC_PKG_DIR}/xccl-${XCCL_BUILD_TYPE}.tgz" --owner=0 --group=0 .
