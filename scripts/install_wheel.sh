#!/usr/bin/env bash
#
# Smart installer for VLM-FO1 wheels
#
# This script detects the local CUDA environment and recommends
# the appropriate wheel variant to install.
#
# Usage:
#   ./scripts/install_wheel.sh [--release VERSION] [--from-local PATH]
#
# Examples:
#   # Install from latest GitHub release
#   ./scripts/install_wheel.sh
#
#   # Install specific version from GitHub
#   ./scripts/install_wheel.sh --release v0.1.0
#
#   # Install from local wheel directory
#   ./scripts/install_wheel.sh --from-local ./dist
#

set -euo pipefail

# Default values
RELEASE_VERSION="latest"
LOCAL_WHEEL_DIR=""
GITHUB_REPO="om-ai-lab/VLM-FO1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

show_help() {
    cat << EOF
VLM-FO1 Smart Wheel Installer

This script detects your CUDA environment and installs the appropriate wheel.

Usage: $0 [OPTIONS]

Options:
    --release VERSION    Install from GitHub release (default: latest)
    --from-local PATH    Install from local wheel directory
    --help               Show this help message

Examples:
    # Auto-detect and install latest release
    $0

    # Install specific version
    $0 --release v0.1.0

    # Install from local build
    $0 --from-local ./dist

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            RELEASE_VERSION="$2"
            shift 2
            ;;
        --from-local)
            LOCAL_WHEEL_DIR="$2"
            shift 2
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

log_info "===================================="
log_info "VLM-FO1 Wheel Installer"
log_info "===================================="

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
log_info "Python version: $PYTHON_VERSION"

# Detect CUDA version
log_step "Detecting CUDA environment..."

CUDA_VERSION=""
RECOMMENDED_VARIANT="cpu"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$CUDA_VERSION" ]]; then
        log_info "CUDA runtime detected: $CUDA_VERSION"

        # Parse major.minor version
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

        # Recommend wheel variant
        if [[ "$CUDA_MAJOR" -eq 11 ]]; then
            RECOMMENDED_VARIANT="cu118"
            log_info "Recommended variant: cu118 (CUDA 11.x compatible)"
        elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
            log_warn "CUDA $CUDA_VERSION detected (12.x+)"
            log_warn "We recommend cu118 for now, which is backward compatible"
            RECOMMENDED_VARIANT="cu118"
        else
            log_warn "CUDA $CUDA_VERSION detected (older than 11.x)"
            log_warn "Falling back to CPU variant"
            RECOMMENDED_VARIANT="cpu"
        fi
    else
        log_warn "nvidia-smi found but could not detect CUDA version"
        RECOMMENDED_VARIANT="cpu"
    fi
else
    log_info "No CUDA detected (nvidia-smi not found)"
    RECOMMENDED_VARIANT="cpu"
fi

log_info "Recommended wheel variant: $RECOMMENDED_VARIANT"

# Determine Python wheel tag
PY_TAG=""
case $PYTHON_VERSION in
    3.9*)
        PY_TAG="cp39-cp39"
        ;;
    3.10*)
        PY_TAG="cp310-cp310"
        ;;
    3.11*)
        PY_TAG="cp311-cp311"
        ;;
    *)
        log_error "Unsupported Python version: $PYTHON_VERSION"
        log_error "Supported versions: 3.9, 3.10, 3.11"
        exit 1
        ;;
esac

log_info "Python wheel tag: $PY_TAG"

# Installation
log_info "===================================="
log_step "Installing VLM-FO1..."

if [[ -n "$LOCAL_WHEEL_DIR" ]]; then
    # Install from local directory
    log_info "Installing from local directory: $LOCAL_WHEEL_DIR"

    WHEEL_PATTERN="$LOCAL_WHEEL_DIR/vlm_fo1-*${RECOMMENDED_VARIANT}*${PY_TAG}*.whl"
    WHEEL_FILE=$(ls $WHEEL_PATTERN 2>/dev/null | head -1 || echo "")

    if [[ -z "$WHEEL_FILE" ]]; then
        log_warn "No matching wheel found for pattern: $WHEEL_PATTERN"
        log_warn "Available wheels:"
        ls -1 "$LOCAL_WHEEL_DIR"/*.whl 2>/dev/null || echo "  (none)"

        log_info "Trying to install any available wheel..."
        WHEEL_FILE=$(ls "$LOCAL_WHEEL_DIR"/*.whl 2>/dev/null | head -1 || echo "")

        if [[ -z "$WHEEL_FILE" ]]; then
            log_error "No wheels found in $LOCAL_WHEEL_DIR"
            exit 1
        fi
    fi

    log_info "Installing wheel: $(basename $WHEEL_FILE)"

    # Install PyTorch first if CUDA variant
    if [[ "$RECOMMENDED_VARIANT" == "cu118" ]]; then
        log_step "Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi

    pip install "$WHEEL_FILE"

else
    # Install from GitHub release
    log_info "Installing from GitHub release: $RELEASE_VERSION"

    BASE_URL="https://github.com/$GITHUB_REPO/releases"

    if [[ "$RELEASE_VERSION" == "latest" ]]; then
        RELEASE_URL="$BASE_URL/latest/download"
    else
        RELEASE_URL="$BASE_URL/download/$RELEASE_VERSION"
    fi

    # Construct wheel filename pattern
    # Example: vlm_fo1-0.1.0+cu118-cp310-cp310-manylinux_2_28_x86_64.whl
    WHEEL_NAME_PATTERN="vlm_fo1-*${RECOMMENDED_VARIANT}*${PY_TAG}*.whl"

    log_info "Download URL base: $RELEASE_URL"
    log_info "Wheel pattern: $WHEEL_NAME_PATTERN"

    # Since we can't easily list GitHub release assets without API,
    # provide manual installation instructions
    echo ""
    log_info "To install from GitHub releases, run:"
    echo ""
    if [[ "$RECOMMENDED_VARIANT" == "cu118" ]]; then
        echo -e "  ${GREEN}# Install PyTorch with CUDA 11.8${NC}"
        echo -e "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        echo ""
    fi
    echo -e "  ${GREEN}# Download and install wheel${NC}"
    echo -e "  wget $RELEASE_URL/$WHEEL_NAME_PATTERN"
    echo -e "  pip install $WHEEL_NAME_PATTERN"
    echo ""
    log_info "Or visit: https://github.com/$GITHUB_REPO/releases"
    exit 0
fi

# Verify installation
log_info "===================================="
log_step "Verifying installation..."

if python3 -c "import vlm_fo1" 2>/dev/null; then
    log_info "✓ VLM-FO1 imported successfully"

    VERSION=$(python3 -c "import vlm_fo1; print(vlm_fo1.__version__)" 2>/dev/null || echo "unknown")
    log_info "Version: $VERSION"

    log_step "Running selfcheck..."
    python3 -m vlm_fo1 || log_warn "Selfcheck reported issues (see above)"

    log_info "===================================="
    log_info "✓ Installation complete!"
    log_info "===================================="
else
    log_error "Installation verification failed"
    log_error "Could not import vlm_fo1"
    exit 1
fi
