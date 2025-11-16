# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import glob
import os
import warnings
from importlib import import_module

from setuptools import find_packages
from setuptools import setup

_EXTENSION_NAME = "MultiScaleDeformableAttention"
_CUSPARSE_HEADER = "cusparse.h"


def _lazy_load_torch():
    """Import torch and its cpp extension utils only when available."""

    try:
        torch = import_module("torch")
        cpp_extension = import_module("torch.utils.cpp_extension")
        return torch, cpp_extension
    except Exception as exc:  # pragma: no cover - build-time diagnostic
        warnings.warn(
            f"PyTorch not detected. Skipping {_EXTENSION_NAME} native build ({exc})."
        )
        return None, None


def _has_cuda_headers(cuda_home):
    if not cuda_home:
        return False
    header_path = os.path.join(cuda_home, "include", _CUSPARSE_HEADER)
    return os.path.isfile(header_path)


def _should_build_cuda(torch_module, cuda_home):
    if not torch_module.cuda.is_available():
        return False
    if not cuda_home:
        warnings.warn(
            "CUDA runtime detected, but CUDA_HOME is unset. Building CPU-only extension."
        )
        return False
    if not _has_cuda_headers(cuda_home):
        warnings.warn(
            f"{_CUSPARSE_HEADER} not found under {cuda_home}. Building CPU-only extension instead."
        )
        return False
    return True


def _build_extensions():
    torch_module, cpp_extension = _lazy_load_torch()
    if torch_module is None or cpp_extension is None:
        return [], {}

    CppExtension = getattr(cpp_extension, "CppExtension", None)
    CUDAExtension = getattr(cpp_extension, "CUDAExtension", None)
    BuildExtension = getattr(cpp_extension, "BuildExtension", None)
    CUDA_HOME = getattr(cpp_extension, "CUDA_HOME", None)

    if CppExtension is None:
        warnings.warn("PyTorch C++ extension helpers missing. Skipping native build.")
        return [], {}

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    define_macros = []
    extra_compile_args = {"cxx": []}
    extension_cls = CppExtension

    if CUDAExtension and source_cuda and _should_build_cuda(torch_module, CUDA_HOME):
        extension_cls = CUDAExtension
        sources += source_cuda
        define_macros.append(("WITH_CUDA", None))
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension_cls(
            _EXTENSION_NAME,
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    cmdclass = {"build_ext": BuildExtension} if BuildExtension else {}
    return ext_modules, cmdclass


ext_modules, cmdclass = _build_extensions()

setup(
    name=_EXTENSION_NAME,
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
