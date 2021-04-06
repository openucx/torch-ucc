#!/bin/bash -eEx
set -o pipefail

echo "INFO: Build UCX"
cd "${TORCH_UCC_SRC_DIR}/ucx"
git checkout "${UCX_BRANCH}"
"${TORCH_UCC_SRC_DIR}/ucx/autogen.sh"
mkdir -p "${TORCH_UCC_SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
cd "${TORCH_UCC_SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
# TODO debug
"${TORCH_UCC_SRC_DIR}/ucx/contrib/configure-release-mt" --with-cuda="${CUDA_HOME}" --prefix="${UCX_INSTALL_DIR}"
#"${TORCH_UCC_SRC_DIR}/ucx/contrib/configure-release-mt" --prefix="${UCX_INSTALL_DIR}"
make -j install
echo "${UCX_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucx.conf
ldconfig
ldconfig -p | grep -i ucx
cd "${UCX_INSTALL_DIR}" && tar cfz "${TORCH_UCC_PKG_DIR}/ucx-${UCX_BUILD_TYPE}.tgz" --owner=0 --group=0 .
