#
# Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

import os
import sys
from setuptools import setup
from torch.utils import cpp_extension

ucc_plugin_dir = os.path.dirname(os.path.abspath(__file__))
ucx_home = os.environ.get("UCX_HOME")
if ucx_home is None:
  print("Couldn't find UCX install dir, please set UCX_HOME env variable")
  sys.exit(1)

ucc_home = os.environ.get("UCC_HOME")
if ucc_home is None:
  print("Couldn't find UCC install dir, please set UCC_HOME env variable")
  sys.exit(1)

plugin_compile_args = []
enable_debug = os.environ.get("ENABLE_DEBUG")
if enable_debug is None or enable_debug == "no":
  print("Release build")
else:
  print("Debug build")
  plugin_compile_args.extend(["-g", "-O0"])

plugin_sources      = ["src/torch_ucc.cpp"]
plugin_include_dirs = ["{}/include/".format(ucc_plugin_dir),
                       "{}/include/".format(ucx_home),
                       "{}/include/".format(ucc_home)]
plugin_library_dirs = ["{}/lib/".format(ucx_home),
                       "{}/lib/".format(ucc_home)]
plugin_libraries    = ["ucp", "uct", "ucm", "ucs", "ucc"]

with_cuda = os.environ.get("WITH_CUDA")
if with_cuda is None or with_cuda == "no":
    print("CUDA support is disabled")
    module = cpp_extension.CppExtension(
        name = "torch_ucc",
        sources = plugin_sources,
        include_dirs = plugin_include_dirs,
        library_dirs = plugin_library_dirs,
        libraries = plugin_libraries,
        extra_compile_args=plugin_compile_args
    )
else:
    print("CUDA support is enabled")
    plugin_compile_args.append("-DUSE_CUDA")
    module = cpp_extension.CUDAExtension(
        name = "torch_ucc",
        sources = plugin_sources,
        include_dirs = plugin_include_dirs,
        library_dirs = plugin_library_dirs,
        libraries = plugin_libraries,
        extra_compile_args=plugin_compile_args
    )
setup(
    name = "torch-ucc",
    version = "1.0.0",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
