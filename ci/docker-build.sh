#!/usr/bin/env bash
#
# Local wheel building script using Docker
#
# This script builds VLM-FO1 wheels locally using Docker containers.
# Supports both CPU and CUDA variants.
#
# Usage:
#   ./ci/docker-build.sh --cuda {cpu|cu118}
#   ./ci/docker-build.sh --cuda cu118 --python 3.10
#   ./ci/docker-build.sh --all  # Build both CPU and CUDA wheels
#
# Options:
#   --cuda VARIANT    CUDA variant to build (cpu, cu118)
#   --python VERSION  Python version to build for (3.9, 3.10, 3.11, or 'all')
#   --all             Build all variants (CPU + CUDA)
#   --output DIR      Output directory for wheels (default: dist/)
#   --no-cache        Build without Docker cache
#   --help            Show this help message
#

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
CUDA_VARIANT=""
PYTHON_VERSION="all"
OUTPUT_DIR="$REPO_ROOT/dist"
BUILD_ALL=false
NO_CACHE_FLAG=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
VLM-FO1 Local Wheel Builder

Usage: $0 [OPTIONS]

Options:
    --cuda VARIANT      CUDA variant: cpu, cu118 (required unless --all is used)
    --python VERSION    Python version: 3.9, 3.10, 3.11, or 'all' (default: all)
    --all               Build all variants (CPU + CUDA cu118)
    --output DIR        Output directory for wheels (default: dist/)
    --no-cache          Build Docker images without cache
    --help              Show this help message

Examples:
    # Build CUDA 11.8 wheels for all Python versions
    $0 --cuda cu118

    # Build CPU-only wheels for Python 3.10
    $0 --cuda cpu --python 3.10

    # Build all variants (CPU + CUDA)
    $0 --all

    # Build with custom output directory
    $0 --cuda cu118 --output /tmp/wheels

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VARIANT="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --all)
            BUILD_ALL=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE_FLAG="--no-cache"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if [[ "$BUILD_ALL" == "false" ]] && [[ -z "$CUDA_VARIANT" ]]; then
    log_error "Either --cuda or --all must be specified"
    show_help
    exit 1
fi

if [[ -n "$CUDA_VARIANT" ]] && [[ ! "$CUDA_VARIANT" =~ ^(cpu|cu118)$ ]]; then
    log_error "Invalid CUDA variant: $CUDA_VARIANT"
    log_error "Valid options: cpu, cu118"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"  # Get absolute path

log_info "===================================="
log_info "VLM-FO1 Wheel Builder"
log_info "===================================="
log_info "Repository: $REPO_ROOT"
log_info "Output directory: $OUTPUT_DIR"
log_info "Python version(s): $PYTHON_VERSION"

# Build function
build_variant() {
    local variant=$1
    local dockerfile=""
    local tag=""

    if [[ "$variant" == "cpu" ]]; then
        dockerfile="$REPO_ROOT/docker/Dockerfile.cpu"
        tag="vlm-fo1-builder:cpu"
    elif [[ "$variant" == "cu118" ]]; then
        dockerfile="$REPO_ROOT/docker/Dockerfile.gpu"
        tag="vlm-fo1-builder:cu118"
    else
        log_error "Unknown variant: $variant"
        return 1
    fi

    log_info "------------------------------------"
    log_info "Building $variant wheels"
    log_info "------------------------------------"
    log_info "Dockerfile: $dockerfile"
    log_info "Tag: $tag"

    # Build Docker image
    log_info "Building Docker image..."
    docker build \
        $NO_CACHE_FLAG \
        -f "$dockerfile" \
        -t "$tag" \
        "$REPO_ROOT"

    # Run container to build wheels
    log_info "Building wheels in container..."
    docker run --rm \
        -v "$OUTPUT_DIR:/output" \
        -e BUILD_CUDA="$variant" \
        "$tag" \
        bash -c 'cp /dist/*.whl /output/ 2>/dev/null && echo "Wheels copied successfully" || echo "Failed to copy wheels"'

    log_info "✓ $variant wheels built successfully"
}

# Build all variants if requested
if [[ "$BUILD_ALL" == "true" ]]; then
    log_info "Building all variants..."
    build_variant "cpu"
    build_variant "cu118"
else
    build_variant "$CUDA_VARIANT"
fi

# Display summary
log_info "===================================="
log_info "Build Complete!"
log_info "===================================="
log_info "Wheels saved to: $OUTPUT_DIR"
log_info ""
log_info "Generated wheels:"
ls -lh "$OUTPUT_DIR"/*.whl 2>/dev/null || log_warn "No wheels found in output directory"

log_info ""
log_info "To test a wheel:"
log_info "  pip install $OUTPUT_DIR/vlm_fo1-*.whl"
log_info "  python -m vlm_fo1"
log_info ""
log_info "To upload to GitHub Releases:"
log_info "  gh release create v0.1.0 $OUTPUT_DIR/*.whl"
log_info "===================================="
