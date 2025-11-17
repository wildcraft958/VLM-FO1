# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
# Enhanced for VLM-FO1: Added BUILD_CUDA environment variable support and improved diagnostics
# ------------------------------------------------------------------------------------------------

import glob
import os
import sys
import warnings
from importlib import import_module

from setuptools import find_packages
from setuptools import setup

_EXTENSION_NAME = "MultiScaleDeformableAttention"
_CUSPARSE_HEADER = "cusparse.h"

# Read BUILD_CUDA environment variable
# Valid values: 'cpu', 'cu118', or unset (auto-detect)
BUILD_CUDA = os.getenv('BUILD_CUDA', 'auto').lower()

print(f"[VLM-FO1 UPN ops] BUILD_CUDA={BUILD_CUDA}", file=sys.stderr)


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


def _get_cuda_home_for_abi(abi):
    """
    Determine CUDA_HOME based on requested ABI.

    Args:
        abi: 'cu118' or other CUDA ABI string

    Returns:
        str: Path to CUDA installation, or None
    """
    # Allow explicit override
    explicit_cuda_home = os.getenv('CUDA_HOME')
    if explicit_cuda_home:
        print(f"[VLM-FO1 UPN ops] Using explicit CUDA_HOME={explicit_cuda_home}", file=sys.stderr)
        return explicit_cuda_home

    # Map ABI to common CUDA installation paths
    if abi == 'cu118':
        candidates = [
            '/usr/local/cuda-11.8',
            '/usr/local/cuda-11',
            '/usr/local/cuda',
        ]
    else:
        candidates = ['/usr/local/cuda']

    for path in candidates:
        if os.path.exists(path):
            print(f"[VLM-FO1 UPN ops] Auto-detected CUDA_HOME={path} for ABI {abi}", file=sys.stderr)
            return path

    return None


def _has_cuda_headers(cuda_home):
    if not cuda_home:
        return False
    header_path = os.path.join(cuda_home, "include", _CUSPARSE_HEADER)
    exists = os.path.isfile(header_path)
    if exists:
        print(f"[VLM-FO1 UPN ops] Found CUDA headers at {header_path}", file=sys.stderr)
    else:
        print(f"[VLM-FO1 UPN ops] CUDA headers NOT found at {header_path}", file=sys.stderr)
    return exists


def _should_build_cuda(torch_module, cuda_home):
    """
    Determine if CUDA extension should be built.

    Respects BUILD_CUDA environment variable:
    - BUILD_CUDA=cpu: Force CPU-only build
    - BUILD_CUDA=cu118: Force CUDA 11.8 build
    - BUILD_CUDA=auto or unset: Auto-detect
    """
    # Explicit CPU-only build requested
    if BUILD_CUDA == 'cpu':
        print("[VLM-FO1 UPN ops] BUILD_CUDA=cpu: Forcing CPU-only build", file=sys.stderr)
        return False

    # Explicit CUDA build requested
    if BUILD_CUDA.startswith('cu'):
        if not cuda_home:
            msg = (
                f"BUILD_CUDA={BUILD_CUDA} requested but CUDA_HOME not found. "
                f"Set CUDA_HOME environment variable or install CUDA toolkit."
            )
            print(f"[VLM-FO1 UPN ops] ERROR: {msg}", file=sys.stderr)
            raise RuntimeError(msg)

        if not _has_cuda_headers(cuda_home):
            msg = (
                f"BUILD_CUDA={BUILD_CUDA} requested but CUDA headers not found at {cuda_home}. "
                f"Install CUDA development headers (cuda-toolkit-11-8 or similar)."
            )
            print(f"[VLM-FO1 UPN ops] ERROR: {msg}", file=sys.stderr)
            raise RuntimeError(msg)

        print(f"[VLM-FO1 UPN ops] Building CUDA extension for {BUILD_CUDA}", file=sys.stderr)
        return True

    # Auto-detect mode
    if not torch_module.cuda.is_available():
        print("[VLM-FO1 UPN ops] Auto-detect: torch.cuda not available, building CPU-only", file=sys.stderr)
        return False

    if not cuda_home:
        warnings.warn(
            "CUDA runtime detected, but CUDA_HOME is unset. Building CPU-only extension. "
            "Set CUDA_HOME or BUILD_CUDA=cu118 to build with CUDA support."
        )
        return False

    if not _has_cuda_headers(cuda_home):
        warnings.warn(
            f"{_CUSPARSE_HEADER} not found under {cuda_home}. Building CPU-only extension instead. "
            f"Install CUDA development headers to enable CUDA extension."
        )
        return False

    print("[VLM-FO1 UPN ops] Auto-detect: Building CUDA extension", file=sys.stderr)
    return True


def _build_extensions():
    torch_module, cpp_extension = _lazy_load_torch()
    if torch_module is None or cpp_extension is None:
        print("[VLM-FO1 UPN ops] PyTorch not available, skipping extension build", file=sys.stderr)
        return [], {}

    CppExtension = getattr(cpp_extension, "CppExtension", None)
    CUDAExtension = getattr(cpp_extension, "CUDAExtension", None)
    BuildExtension = getattr(cpp_extension, "BuildExtension", None)

    # Get CUDA_HOME: prefer explicit, then ABI-specific, then torch default
    if BUILD_CUDA.startswith('cu'):
        CUDA_HOME = _get_cuda_home_for_abi(BUILD_CUDA)
    else:
        CUDA_HOME = getattr(cpp_extension, "CUDA_HOME", None)

    if CppExtension is None:
        warnings.warn("PyTorch C++ extension helpers missing. Skipping native build.")
        return [], {}

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    # Determine if we should build CUDA extension
    should_build_cuda = (
        CUDAExtension
        and source_cuda
        and _should_build_cuda(torch_module, CUDA_HOME)
    )

    if not should_build_cuda:
        print(
            "[VLM-FO1 UPN ops] CUDA extension build skipped. CPU fallback will be used at runtime.",
            file=sys.stderr
        )
        return [], {}

    # Build CUDA extension
    sources = main_file + source_cpu + source_cuda
    define_macros = [("WITH_CUDA", None)]

    # CUDA architecture list for cu118 (CUDA 11.8)
    # Covers common GPUs: Pascal (6.0, 6.1), Volta (7.0), Turing (7.5), Ampere (8.0, 8.6), Ada (8.9)
    nvcc_args = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    if BUILD_CUDA == 'cu118':
        # Explicit architecture list for CUDA 11.8
        nvcc_args.extend([
            "-gencode=arch=compute_60,code=sm_60",  # Pascal
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",  # Volta
            "-gencode=arch=compute_75,code=sm_75",  # Turing
            "-gencode=arch=compute_80,code=sm_80",  # Ampere
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",  # Ada (if supported)
        ])

    extra_compile_args = {
        "cxx": [],
        "nvcc": nvcc_args,
    }

    print(f"[VLM-FO1 UPN ops] Building CUDA extension with sources: {len(sources)} files", file=sys.stderr)
    print(f"[VLM-FO1 UPN ops] CUDA_HOME: {CUDA_HOME}", file=sys.stderr)

    ext_modules = [
        CUDAExtension(
            _EXTENSION_NAME,
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    cmdclass = {"build_ext": BuildExtension} if BuildExtension else {}
    print(f"[VLM-FO1 UPN ops] Extension build configured successfully", file=sys.stderr)
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
