#!/bin/bash -eEx
set -o pipefail

echo "INFO: Build XCCL"
XCCL_SRC_DIR="${TORCH_UCC_SRC_DIR}/xccl"
cd "${XCCL_SRC_DIR}"
"${XCCL_SRC_DIR}/autogen.sh"
mkdir -p "${XCCL_SRC_DIR}/build-${XCCL_BUILD_TYPE}"
cd "${XCCL_SRC_DIR}/build-${XCCL_BUILD_TYPE}"
# TODO enable CUDA (compilation failed)
#"${XCCL_SRC_DIR}/configure" --with-ucx="${UCX_INSTALL_DIR}" --prefix="${XCCL_INSTALL_DIR}" --enable-debug
"${XCCL_SRC_DIR}/configure" --with-cuda="${CUDA_HOME}" --with-ucx="${UCX_INSTALL_DIR}" \
    --prefix="${XCCL_INSTALL_DIR}" --enable-debug
make -j install
echo "${XCCL_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/xccl.conf
ldconfig
ldconfig -p | grep -i libxccl
make -C test
cd "${XCCL_INSTALL_DIR}" && tar cfz "${TORCH_UCC_PKG_DIR}/xccl-${XCCL_BUILD_TYPE}.tgz" --owner=0 --group=0 .
