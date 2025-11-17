# GPU Development Guide for VLM-FO1

This guide explains how to build, test, and deploy VLM-FO1 with reliable GPU inference support.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Building from Source](#building-from-source)
- [Testing on GPU](#testing-on-gpu)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Check Your Environment

First, diagnose your CUDA setup:

```bash
python tools/diagnose_cuda.py
```

This will tell you exactly which wheel variant to install.

### 2. Install VLM-FO1

**For CUDA 11.8 systems:**
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install VLM-FO1 (from source with uv)
uv pip install -e ".[demo]"
```

**For CPU-only systems:**
```bash
# Install PyTorch CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install VLM-FO1
BUILD_CUDA=cpu uv pip install -e ".[demo]"
```

### 3. Verify Installation

```bash
python -m vlm_fo1
```

This runs a selfcheck that validates:
- GPU availability
- CUDA driver and runtime versions
- PyTorch CUDA compatibility
- UPN extension loading
- ABI compatibility

---

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install VLM-FO1 with all dependencies
uv pip install -e ".[all]"
```

### Using pip

```bash
# Install core dependencies
pip install -e .

# Or with specific extras
pip install -e ".[demo]"      # Demo and visualization
pip install -e ".[upn]"        # UPN detector
pip install -e ".[dev]"        # Development tools
pip install -e ".[all]"        # Everything
```

### Pre-built Wheels

Download from [GitHub Releases](https://github.com/om-ai-lab/VLM-FO1/releases):

```bash
# CUDA 11.8 wheel
wget https://github.com/om-ai-lab/VLM-FO1/releases/download/v0.1.0/vlm_fo1-0.1.0+cu118-cp310-cp310-manylinux_2_28_x86_64.whl
pip install vlm_fo1-0.1.0+cu118-cp310-cp310-manylinux_2_28_x86_64.whl

# Or use the smart installer
./scripts/install_wheel.sh --release v0.1.0
```

---

## Building from Source

### Local Development Build

```bash
# For CUDA 11.8
BUILD_CUDA=cu118 uv pip install -e ".[dev]"

# For CPU only
BUILD_CUDA=cpu uv pip install -e ".[dev]"
```

### Building Wheels with Docker

Build production wheels for all Python versions:

```bash
# Build CUDA 11.8 wheels
./ci/docker-build.sh --cuda cu118

# Build CPU wheels
./ci/docker-build.sh --cuda cpu

# Build all variants
./ci/docker-build.sh --all
```

Wheels will be in `dist/` directory.

### Manual Wheel Building

```bash
# Install build dependencies
pip install build wheel setuptools>=68 setuptools-scm>=8

# Build wheel
BUILD_CUDA=cu118 python -m build --wheel

# Repair for manylinux (Linux only)
pip install auditwheel
auditwheel repair dist/*.whl --plat manylinux_2_28_x86_64 -w dist/
```

---

## Testing on GPU

### Run Smoke Tests

```bash
# Full GPU smoke test suite
pytest tests/e2e_gpu_smoke.py -v

# Skip GPU tests (CPU only)
pytest tests/e2e_gpu_smoke.py -v -m "not gpu"
```

### Benchmark in Docker

Run comprehensive GPU benchmarks in an isolated environment:

```bash
# CUDA 11.8 benchmark
./bench/run_gpu_smoke.sh --cuda cu118

# CPU benchmark
./bench/run_gpu_smoke.sh --cuda cpu
```

Results are saved to `bench/results/`.

### Test Individual Features

```bash
# Test model loading
python -c "from vlm_fo1.model.builder import load_pretrained_model; print('✓ Model loading works')"

# Test UPN extension
python -c "from detect_tools.upn.ops import MultiScaleDeformableAttention; print('✓ UPN extension works')"

# Test backend diagnostics
python -c "from vlm_fo1._backend import info; import json; print(json.dumps(info(), indent=2))"
```

---

## Deployment

### Hugging Face Spaces

VLM-FO1 can be deployed to [Hugging Face Spaces](https://huggingface.co/spaces) with GPU support.

**1. Create `app.py`:**

```python
import gradio as gr
import os

# Set to use pre-built wheel or install from source
os.environ['BUILD_CUDA'] = 'cu118'

# Your Gradio app code here (see demo/gradio_demo.py)
from demo.gradio_demo import create_demo

demo = create_demo()

if __name__ == "__main__":
    demo.launch()
```

**2. Create `requirements.txt`:**

```txt
torch>=2.1,<2.7
torchvision>=0.16
transformers>=4.45
gradio>=4.0
git+https://github.com/om-ai-lab/VLM-FO1.git
```

**3. Configure Space:**

- Set hardware to **GPU T4** or higher
- Python version: **3.10**
- Add your model files to the Space or load from Hugging Face Hub

**4. Deploy:**

Push your files to the Space repository. The build will automatically install dependencies and start the Gradio app.

### Modal

For deployment on [Modal](https://modal.com/):

```python
import modal

app = modal.App("vlm-fo1")

# Define CUDA image
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("vlm-fo1[demo]")
)

@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=600,
)
def run_inference(image_path, query):
    from vlm_fo1.model.builder import load_pretrained_model
    from vlm_fo1.mm_utils import prepare_inputs
    from vlm_fo1.task_templates import OD_template
    import torch

    # Load model
    tokenizer, model, image_processors = load_pretrained_model(
        "omlab/VLM-FO1_Qwen2.5-VL-3B-v01",
        device="cuda"
    )

    # Run inference
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": OD_template.format(query)},
        ],
        "bbox_list": [[0, 0, 100, 100]],  # Example bbox
    }]

    generation_kwargs = prepare_inputs(
        "omlab/VLM-FO1_Qwen2.5-VL-3B-v01",
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=512,
        temperature=0.0,
    )

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    return tokenizer.decode(output_ids[0])

@app.local_entrypoint()
def main():
    result = run_inference.remote("path/to/image.jpg", "orange")
    print(result)
```

Deploy with:
```bash
modal deploy vlm_fo1_app.py
```

### Docker Deployment

**CPU-only:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    BUILD_CUDA=cpu pip install -e ".[demo]"

CMD ["python", "demo/gradio_demo.py"]
```

**CUDA 11.8:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    BUILD_CUDA=cu118 pip install -e ".[demo]"

CMD ["python3", "demo/gradio_demo.py"]
```

Build and run:
```bash
docker build -t vlm-fo1:cu118 .
docker run --gpus all -p 7860:7860 vlm-fo1:cu118
```

---

## Troubleshooting

### Common Issues

#### 1. "CUDA extension failed to load"

**Symptoms:**
```
ERROR: Wheel built for cu118 (CUDA 11.8). System driver/runtime reports CUDA 12.x.
```

**Solution:**
```bash
# Check your CUDA version
python tools/diagnose_cuda.py

# Install matching PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Reinstall VLM-FO1
BUILD_CUDA=cu118 pip install --force-reinstall -e .
```

#### 2. "UPN extension using CPU fallback"

**Symptoms:**
```
[VLM-FO1 UPN ops] CUDA extension build skipped. CPU fallback will be used at runtime.
```

**Cause:** CUDA headers not found during build.

**Solution:**
```bash
# Install CUDA toolkit headers (Ubuntu/Debian)
sudo apt-get install cuda-toolkit-11-8

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-11.8

# Rebuild extension
cd detect_tools/upn/ops
BUILD_CUDA=cu118 pip install -v -e . --no-build-isolation
```

#### 3. "ModuleNotFoundError: No module named 'vlm_fo1'"

**Solution:**
```bash
# Install in editable mode from repo root
pip install -e .

# Or if using uv
uv pip install -e .
```

#### 4. Model Loading Fails on Hugging Face Spaces

**Symptoms:**
```
HfApiError: 401 Client Error: Unauthorized
```

**Solution:**
- Add a Hugging Face token to your Space secrets
- Or make sure model repository is public
- Use `huggingface-cli login` for local development

#### 5. Out of Memory (OOM) on GPU

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
```python
# Use quantization (8-bit or 4-bit)
tokenizer, model, image_processors = load_pretrained_model(
    model_path,
    load_8bit=True,  # or load_4bit=True
    device="cuda"
)

# Reduce batch size / max tokens
generation_kwargs = prepare_inputs(
    ...,
    max_tokens=256,  # Reduce from 4096
)

# Use gradient checkpointing (for training)
model.gradient_checkpointing_enable()
```

### Driver Compatibility

| CUDA Version | Minimum Driver | Recommended |
|--------------|----------------|-------------|
| 11.8         | 520.61.05      | 520.x+      |
| 12.1         | 530.30.02      | 535.x+      |

Check your driver version:
```bash
nvidia-smi
```

Upgrade drivers:
```bash
# Ubuntu/Debian
sudo ubuntu-drivers install

# Or download from NVIDIA:
# https://www.nvidia.com/Download/index.aspx
```

### Getting Help

1. **Run diagnostics:**
   ```bash
   python -m vlm_fo1 > diagnostics.txt
   python tools/diagnose_cuda.py >> diagnostics.txt
   ```

2. **Check logs:**
   - Build logs: Look for `[VLM-FO1 UPN ops]` messages during install
   - Runtime logs: Check stderr for CUDA errors

3. **Open an issue:**
   - Include `diagnostics.txt`
   - Specify your environment (OS, Python version, GPU model)
   - Include full error traceback

   https://github.com/om-ai-lab/VLM-FO1/issues

---

## Developer Notes

### Project Structure

```
VLM-FO1/
├── vlm_fo1/                 # Main package
│   ├── _backend.py          # GPU diagnostics
│   ├── __main__.py          # Selfcheck CLI
│   ├── model/               # Model architecture
│   └── mm_utils.py          # Utilities
├── detect_tools/upn/        # UPN detector
│   └── ops/                 # CUDA extension
├── tests/                   # Test suite
│   └── e2e_gpu_smoke.py     # GPU smoke tests
├── docker/                  # Docker build files
├── ci/                      # Build scripts
├── bench/                   # Benchmarking tools
└── tools/                   # Diagnostic utilities
```

### Build Variables

- `BUILD_CUDA`: Controls CUDA extension build (`cpu`, `cu118`, `auto`)
- `CUDA_HOME`: Path to CUDA toolkit (auto-detected if not set)
- `VLM_FO1_SKIP_VALIDATION`: Skip runtime validation on import (`1` to skip)
- `VLM_FO1_TEST_MODEL`: Model path for tests (default: HuggingFace)

### CI/CD

GitHub Actions automatically builds wheels on:
- Push to `main` branch
- Pull requests
- Tag pushes (`v*`)

Wheels are uploaded to GitHub Releases for tags.

---

## License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.
