"""
Runtime GPU diagnostics and backend validation for VLM-FO1.

This module detects GPU availability, CUDA driver versions, torch CUDA compatibility,
and validates wheel ABI compatibility with the runtime environment.
"""

import ctypes.util
import json
import os
import platform
import subprocess
import sys
import warnings
from typing import Dict, Optional, Any


def _get_wheel_abi() -> str:
    """
    Extract the CUDA ABI suffix from the installed wheel version.

    Returns:
        str: One of 'cpu', 'cu118', or 'unknown'
    """
    try:
        import vlm_fo1
        version = getattr(vlm_fo1, '__version__', '0.1.0')

        # Check for ABI suffix in version (e.g., '0.1.0+cu118')
        if '+cu118' in version:
            return 'cu118'
        elif '+cpu' in version:
            return 'cpu'
        else:
            # No explicit ABI tag; assume CPU if no CUDA found during build
            return 'unknown'
    except ImportError:
        return 'unknown'


def _get_cuda_driver_version() -> Optional[str]:
    """
    Query CUDA driver version using nvidia-smi.

    Returns:
        Optional[str]: Driver version string (e.g., '520.61.05') or None if unavailable
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=5
        )
        return result.strip().split('\n')[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_cuda_runtime_version() -> Optional[str]:
    """
    Query CUDA runtime version using nvidia-smi.

    Returns:
        Optional[str]: CUDA version string (e.g., '11.8') or None if unavailable
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=5
        )
        return result.strip().split('\n')[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _find_cudart_library() -> Optional[str]:
    """
    Locate libcudart shared library on the system.

    Returns:
        Optional[str]: Path to libcudart or None if not found
    """
    # Try ctypes util first
    lib = ctypes.util.find_library('cudart')
    if lib:
        return lib

    # Manual search in common CUDA locations
    cuda_lib_paths = [
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
    ]

    for path in cuda_lib_paths:
        for filename in ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0']:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path

    return None


def _check_torch_cuda() -> Dict[str, Any]:
    """
    Check PyTorch CUDA availability and version.

    Returns:
        dict: {'available': bool, 'version': Optional[str], 'error': Optional[str]}
    """
    try:
        import torch

        return {
            'available': torch.cuda.is_available(),
            'version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'error': None
        }
    except ImportError as e:
        return {
            'available': False,
            'version': None,
            'device_count': 0,
            'error': f'PyTorch not installed: {e}'
        }


def _check_upn_extension() -> Dict[str, Any]:
    """
    Attempt to load the UPN MultiScaleDeformableAttention extension.

    Returns:
        dict: {'loaded': bool, 'backend': str, 'error': Optional[str]}
    """
    try:
        from detect_tools.upn.ops import MultiScaleDeformableAttention

        # Check if it's the CUDA or CPU fallback version
        module_file = MultiScaleDeformableAttention.__module__
        is_fallback = 'cpu_fallback' in module_file

        return {
            'loaded': True,
            'backend': 'cpu_fallback' if is_fallback else 'cuda',
            'error': None
        }
    except ImportError as e:
        return {
            'loaded': False,
            'backend': 'none',
            'error': str(e)
        }
    except Exception as e:
        return {
            'loaded': False,
            'backend': 'none',
            'error': f'Extension load error: {e}'
        }


def info() -> Dict[str, Any]:
    """
    Gather comprehensive runtime diagnostics.

    Returns:
        dict: Complete diagnostic information including:
            - python: Python version
            - platform: OS platform
            - wheel_abi: Installed wheel CUDA ABI
            - driver_version: NVIDIA driver version
            - cuda_runtime_version: CUDA runtime version
            - cudart_library: Path to libcudart.so
            - torch: PyTorch CUDA info
            - upn_extension: UPN extension load status
            - cuda_available: Overall CUDA availability
            - abi_compatible: Whether wheel ABI matches runtime
    """
    wheel_abi = _get_wheel_abi()
    driver_version = _get_cuda_driver_version()
    cuda_runtime = _get_cuda_runtime_version()
    cudart_lib = _find_cudart_library()
    torch_info = _check_torch_cuda()
    extension_info = _check_upn_extension()

    # Determine overall CUDA availability
    cuda_available = (
        driver_version is not None and
        torch_info['available'] and
        cudart_lib is not None
    )

    # Check ABI compatibility
    abi_compatible = True
    abi_warnings = []

    if wheel_abi == 'cu118':
        # cu118 wheel expects CUDA 11.x runtime
        if torch_info['version'] and not torch_info['version'].startswith('11.'):
            abi_compatible = False
            abi_warnings.append(
                f"Wheel ABI is cu118 but torch.version.cuda={torch_info['version']}. "
                "Expected CUDA 11.x."
            )

        if cuda_runtime and not cuda_runtime.startswith('11.'):
            abi_compatible = False
            abi_warnings.append(
                f"Wheel ABI is cu118 but system CUDA runtime is {cuda_runtime}. "
                "Expected CUDA 11.x."
            )

    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'wheel_abi': wheel_abi,
        'driver_version': driver_version,
        'cuda_runtime_version': cuda_runtime,
        'cudart_library': cudart_lib,
        'torch': torch_info,
        'upn_extension': extension_info,
        'cuda_available': cuda_available,
        'abi_compatible': abi_compatible,
        'abi_warnings': abi_warnings if abi_warnings else None
    }


def validate() -> None:
    """
    Validate runtime environment and raise descriptive errors on mismatch.

    Raises:
        RuntimeError: If critical ABI mismatches or missing dependencies are detected
    """
    diagnostics = info()

    wheel_abi = diagnostics['wheel_abi']

    # If it's a CPU wheel, no validation needed
    if wheel_abi == 'cpu':
        return

    # If it's a CUDA wheel, validate environment
    if wheel_abi == 'cu118':
        if not diagnostics['cuda_available']:
            msg = (
                f"ERROR: Wheel built for cu118 (CUDA 11.8), but CUDA is not available on this system.\n"
                f"  Driver version: {diagnostics['driver_version'] or 'NOT FOUND'}\n"
                f"  Torch CUDA: {diagnostics['torch']['available']}\n"
                f"  libcudart: {diagnostics['cudart_library'] or 'NOT FOUND'}\n\n"
                f"Remediation:\n"
                f"  1. Install CUDA 11.8 drivers and runtime, OR\n"
                f"  2. Install PyTorch with CUDA 11.8 support, OR\n"
                f"  3. Install the CPU-only wheel instead.\n"
                f"See CONTRIBUTING_GPU.md for details."
            )
            raise RuntimeError(msg)

        if not diagnostics['abi_compatible']:
            warnings_str = '\n  '.join(diagnostics['abi_warnings'] or [])
            msg = (
                f"ERROR: Wheel ABI mismatch detected.\n"
                f"  Wheel ABI: {wheel_abi}\n"
                f"  Warnings:\n  {warnings_str}\n\n"
                f"Remediation:\n"
                f"  1. Install CUDA 11.8-compatible PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                f"  2. Ensure system CUDA runtime is 11.x\n"
                f"  3. Or install a different wheel variant matching your system.\n"
                f"See CONTRIBUTING_GPU.md#driver-compatibility for details."
            )
            raise RuntimeError(msg)

    # Unknown ABI - warn but don't fail
    if wheel_abi == 'unknown':
        warnings.warn(
            "Unable to determine wheel CUDA ABI. Proceeding with caution. "
            "Run 'python -m vlm_fo1' for diagnostics."
        )


def selfcheck() -> int:
    """
    Run self-check diagnostics and print results.

    Returns:
        int: Exit code (0=success, 1=no CUDA, 2=ABI mismatch, 3=extension failed)
    """
    diagnostics = info()

    print(json.dumps(diagnostics, indent=2))

    # Determine exit code
    if diagnostics['wheel_abi'] == 'cpu':
        # CPU wheel is always successful if it loads
        return 0

    if not diagnostics['cuda_available']:
        print("\n❌ CUDA not available", file=sys.stderr)
        return 1

    if not diagnostics['abi_compatible']:
        print("\n❌ ABI mismatch detected", file=sys.stderr)
        if diagnostics['abi_warnings']:
            for warning in diagnostics['abi_warnings']:
                print(f"  - {warning}", file=sys.stderr)
        return 2

    if not diagnostics['upn_extension']['loaded']:
        print(f"\n⚠️  UPN extension failed to load: {diagnostics['upn_extension']['error']}", file=sys.stderr)
        return 3

    print("\n✅ All checks passed", file=sys.stderr)
    return 0


# Automatically validate on import (can be disabled with env var)
if os.getenv('VLM_FO1_SKIP_VALIDATION') != '1':
    try:
        validate()
    except RuntimeError as e:
        # Don't fail import, just warn
        warnings.warn(str(e), stacklevel=2)
