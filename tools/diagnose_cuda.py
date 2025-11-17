#!/usr/bin/env python3
"""
CUDA diagnostic utility for VLM-FO1.

This standalone script diagnoses CUDA environment and recommends
the correct wheel variant to install. Can be run before installing VLM-FO1.

Usage:
    python tools/diagnose_cuda.py
    python tools/diagnose_cuda.py --json
    python tools/diagnose_cuda.py --check-wheel vlm_fo1-0.1.0+cu118-cp310-cp310-linux_x86_64.whl
"""

import argparse
import ctypes.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_nvidia_smi():
    """Check if nvidia-smi is available and get driver info."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version,cuda_version,name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=5
        )
        lines = result.strip().split('\n')
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpus.append({
                    'driver_version': parts[0],
                    'cuda_version': parts[1],
                    'name': parts[2]
                })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_cuda_toolkit():
    """Check for CUDA toolkit installation."""
    cuda_homes = [
        os.getenv('CUDA_HOME'),
        '/usr/local/cuda',
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-11',
    ]

    found = []
    for cuda_home in cuda_homes:
        if cuda_home and os.path.exists(cuda_home):
            nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.check_output(
                        [nvcc_path, '--version'],
                        stderr=subprocess.DEVNULL,
                        universal_newlines=True,
                        timeout=5
                    )
                    # Parse version from output
                    for line in result.split('\n'):
                        if 'release' in line.lower():
                            version = line.split('release')[-1].split(',')[0].strip()
                            found.append({
                                'path': cuda_home,
                                'version': version
                            })
                            break
                except:
                    pass

    return found if found else None


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        return {
            'installed': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': [
                torch.cuda.get_device_properties(i).name
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else []
        }
    except ImportError:
        return {
            'installed': False,
            'error': 'PyTorch not installed'
        }


def check_cudart_library():
    """Find libcudart shared library."""
    lib = ctypes.util.find_library('cudart')
    if lib:
        return lib

    # Manual search
    search_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-11.8/lib64',
        '/usr/lib/x86_64-linux-gnu',
    ]

    for path in search_paths:
        for libname in ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0']:
            full_path = os.path.join(path, libname)
            if os.path.exists(full_path):
                return full_path

    return None


def recommend_variant(diagnostics):
    """Recommend wheel variant based on diagnostics."""
    gpus = diagnostics.get('nvidia_smi')
    pytorch = diagnostics.get('pytorch', {})

    if not gpus:
        return {
            'variant': 'cpu',
            'reason': 'No NVIDIA GPU detected',
            'install_command': 'pip install vlm-fo1 (CPU variant)'
        }

    # Get CUDA version from GPU
    cuda_version = gpus[0]['cuda_version'] if gpus else None

    if cuda_version:
        major = cuda_version.split('.')[0]

        if major == '11':
            return {
                'variant': 'cu118',
                'reason': f'CUDA {cuda_version} detected (11.x series)',
                'install_command': (
                    'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n'
                    'pip install vlm-fo1+cu118'
                ),
                'pytorch_check': pytorch.get('cuda_version', 'not installed') if pytorch.get('installed') else 'not installed'
            }
        elif int(major) >= 12:
            return {
                'variant': 'cu118',
                'reason': f'CUDA {cuda_version} detected (12.x+, using cu118 for compatibility)',
                'install_command': (
                    'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n'
                    'pip install vlm-fo1+cu118'
                ),
                'warning': 'CUDA 12.x detected but cu118 wheels are backward compatible',
                'pytorch_check': pytorch.get('cuda_version', 'not installed') if pytorch.get('installed') else 'not installed'
            }

    return {
        'variant': 'cpu',
        'reason': 'Could not determine CUDA version reliably',
        'install_command': 'pip install vlm-fo1 (CPU variant)'
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose CUDA environment for VLM-FO1')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--check-wheel', metavar='WHEEL', help='Check compatibility of a specific wheel file')
    args = parser.parse_args()

    # Gather diagnostics
    diagnostics = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'python_version': platform.python_version(),
        },
        'nvidia_smi': check_nvidia_smi(),
        'cuda_toolkit': check_cuda_toolkit(),
        'cudart_library': check_cudart_library(),
        'pytorch': check_pytorch(),
    }

    # Get recommendation
    recommendation = recommend_variant(diagnostics)
    diagnostics['recommendation'] = recommendation

    # Check specific wheel if requested
    if args.check_wheel:
        wheel_path = Path(args.check_wheel)
        wheel_name = wheel_path.name

        # Parse wheel ABI
        wheel_abi = 'cpu'
        if '+cu118' in wheel_name:
            wheel_abi = 'cu118'

        cuda_version = diagnostics['nvidia_smi'][0]['cuda_version'] if diagnostics['nvidia_smi'] else None
        compatible = False
        reason = ""

        if wheel_abi == 'cpu':
            compatible = True
            reason = "CPU wheel works on any system"
        elif wheel_abi == 'cu118':
            if cuda_version and cuda_version.startswith('11.'):
                compatible = True
                reason = f"CUDA {cuda_version} is compatible with cu118 wheel"
            else:
                compatible = False
                reason = f"CUDA {cuda_version or 'not found'} may not be compatible with cu118 wheel (expects CUDA 11.x)"

        diagnostics['wheel_check'] = {
            'wheel': wheel_name,
            'wheel_abi': wheel_abi,
            'compatible': compatible,
            'reason': reason
        }

    # Output
    if args.json:
        print(json.dumps(diagnostics, indent=2))
    else:
        print("=" * 60)
        print("VLM-FO1 CUDA Environment Diagnostic")
        print("=" * 60)
        print()

        print("Platform:")
        print(f"  OS: {diagnostics['platform']['system']} {diagnostics['platform']['release']}")
        print(f"  Python: {diagnostics['platform']['python_version']}")
        print()

        print("NVIDIA GPUs:")
        if diagnostics['nvidia_smi']:
            for i, gpu in enumerate(diagnostics['nvidia_smi']):
                print(f"  GPU {i}: {gpu['name']}")
                print(f"    Driver: {gpu['driver_version']}")
                print(f"    CUDA: {gpu['cuda_version']}")
        else:
            print("  Not detected (nvidia-smi not found)")
        print()

        print("CUDA Toolkit:")
        if diagnostics['cuda_toolkit']:
            for toolkit in diagnostics['cuda_toolkit']:
                print(f"  Path: {toolkit['path']}")
                print(f"  Version: {toolkit['version']}")
        else:
            print("  Not found")
        print()

        print("CUDA Runtime Library:")
        if diagnostics['cudart_library']:
            print(f"  Found: {diagnostics['cudart_library']}")
        else:
            print("  Not found")
        print()

        print("PyTorch:")
        pytorch = diagnostics['pytorch']
        if pytorch.get('installed'):
            print(f"  Version: {pytorch['version']}")
            print(f"  CUDA available: {pytorch['cuda_available']}")
            if pytorch['cuda_available']:
                print(f"  CUDA version: {pytorch['cuda_version']}")
                print(f"  GPU count: {pytorch['device_count']}")
                for i, device in enumerate(pytorch['devices']):
                    print(f"    GPU {i}: {device}")
        else:
            print(f"  Not installed")
        print()

        print("=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        rec = recommendation
        print(f"Recommended variant: {rec['variant']}")
        print(f"Reason: {rec['reason']}")
        if 'warning' in rec:
            print(f"⚠️  Warning: {rec['warning']}")
        print()
        print("Installation command:")
        print(rec['install_command'])
        print()

        if 'pytorch_check' in rec and rec['pytorch_check'] != 'not installed':
            cuda_match = rec['pytorch_check'].startswith('11.') if rec['variant'] == 'cu118' else True
            if cuda_match:
                print(f"✓ PyTorch CUDA version ({rec['pytorch_check']}) matches recommendation")
            else:
                print(f"⚠️  PyTorch CUDA version ({rec['pytorch_check']}) may not match {rec['variant']}")
                print(f"   Reinstall PyTorch with: pip install torch --index-url https://download.pytorch.org/whl/cu118")

        if args.check_wheel and 'wheel_check' in diagnostics:
            print()
            print("=" * 60)
            print("WHEEL COMPATIBILITY CHECK")
            print("=" * 60)
            wc = diagnostics['wheel_check']
            print(f"Wheel: {wc['wheel']}")
            print(f"Wheel ABI: {wc['wheel_abi']}")
            print(f"Compatible: {'✓ Yes' if wc['compatible'] else '✗ No'}")
            print(f"Reason: {wc['reason']}")

        print()
        print("=" * 60)


if __name__ == '__main__':
    main()
