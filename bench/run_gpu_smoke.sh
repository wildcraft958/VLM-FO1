#!/usr/bin/env bash
#
# Run GPU smoke test in Docker container and output JSON report
#
# Usage:
#   ./bench/run_gpu_smoke.sh [--cuda {cu118|cpu}] [--image nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04]
#
# Example:
#   ./bench/run_gpu_smoke.sh --cuda cu118
#   ./bench/run_gpu_smoke.sh --cuda cpu --image python:3.10-slim
#

set -euo pipefail

# Default values
CUDA_VARIANT="cu118"
DOCKER_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
PYTHON_VERSION="3.10"
OUTPUT_DIR="bench/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VARIANT="$2"
            shift 2
            ;;
        --image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cuda {cu118|cpu}] [--image IMAGE] [--python VERSION]"
            exit 1
            ;;
    esac
done

# Set appropriate Docker image for CPU variant
if [[ "$CUDA_VARIANT" == "cpu" ]]; then
    DOCKER_IMAGE="python:${PYTHON_VERSION}-slim"
    GPU_FLAGS=""
else
    GPU_FLAGS="--gpus all"
fi

echo "==================================="
echo "VLM-FO1 GPU Smoke Test Benchmark"
echo "==================================="
echo "CUDA variant: $CUDA_VARIANT"
echo "Docker image: $DOCKER_IMAGE"
echo "Python version: $PYTHON_VERSION"
echo "GPU flags: ${GPU_FLAGS:-none (CPU only)}"
echo "==================================="
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run smoke test in Docker
echo "Starting Docker container..."
docker run --rm \
    ${GPU_FLAGS} \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -e BUILD_CUDA="$CUDA_VARIANT" \
    -e PYTHONUNBUFFERED=1 \
    "$DOCKER_IMAGE" \
    bash -c '
        set -euo pipefail

        echo "Installing system dependencies..."
        if command -v apt-get &> /dev/null; then
            apt-get update -qq
            apt-get install -y -qq git build-essential > /dev/null
        elif command -v yum &> /dev/null; then
            yum install -y -q git gcc gcc-c++ make > /dev/null
        fi

        echo "Installing Python and pip..."
        if ! command -v python3 &> /dev/null; then
            apt-get install -y -qq python3 python3-pip python3-venv > /dev/null
        fi

        echo "Creating virtual environment..."
        python3 -m venv /tmp/venv
        source /tmp/venv/bin/activate

        echo "Upgrading pip..."
        pip install --quiet --upgrade pip setuptools wheel

        echo "Installing VLM-FO1 package..."
        if [[ "$BUILD_CUDA" == "cpu" ]]; then
            pip install --quiet -e ".[dev]"
        else
            # Install torch with CUDA support first
            pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
            pip install --quiet -e ".[dev]"
        fi

        echo "Running selfcheck..."
        python -m vlm_fo1 || true

        echo ""
        echo "Running GPU smoke tests..."
        if [[ "$BUILD_CUDA" == "cpu" ]]; then
            echo "CPU mode - skipping GPU tests"
            pytest tests/e2e_gpu_smoke.py -v -m "not gpu" --tb=short || true
        else
            pytest tests/e2e_gpu_smoke.py -v --tb=short -o junit_family=xunit2 --junitxml=/workspace/bench/results/test-results-'"${TIMESTAMP}"'.xml || true
        fi

        echo ""
        echo "Generating JSON report..."
        python3 << "EOF"
import json
import sys
from datetime import datetime

try:
    from vlm_fo1._backend import info as backend_info
    diag = backend_info()
except Exception as e:
    diag = {"error": str(e)}

report = {
    "timestamp": datetime.now().isoformat(),
    "cuda_variant": "'$BUILD_CUDA'",
    "docker_image": "'$DOCKER_IMAGE'",
    "diagnostics": diag,
    "test_status": "completed"
}

print(json.dumps(report, indent=2))

with open("/workspace/bench/results/report-'"${TIMESTAMP}"'.json", "w") as f:
    json.dump(report, f, indent=2)
EOF

        echo ""
        echo "Benchmark complete!"
    '

# Display summary
echo ""
echo "==================================="
echo "Benchmark Results"
echo "==================================="

LATEST_REPORT=$(ls -t "$OUTPUT_DIR"/report-*.json 2>/dev/null | head -1 || echo "")
if [[ -n "$LATEST_REPORT" ]]; then
    echo "Report saved to: $LATEST_REPORT"
    echo ""
    cat "$LATEST_REPORT"
else
    echo "No report generated"
fi

echo ""
echo "Test results saved in: $OUTPUT_DIR"
echo "==================================="
