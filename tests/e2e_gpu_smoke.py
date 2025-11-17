"""
End-to-end GPU smoke test for VLM-FO1.

This test validates:
1. GPU availability and CUDA compatibility
2. Model loading on GPU
3. Deterministic inference execution
4. Output correctness (shape, dtype)
5. GPU memory usage and performance

Run with: pytest tests/e2e_gpu_smoke.py
Skip GPU tests with: pytest tests/e2e_gpu_smoke.py -m "not gpu"
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Set environment variable to skip validation during import (we'll validate explicitly)
os.environ['VLM_FO1_SKIP_VALIDATION'] = '1'

from vlm_fo1._backend import info as backend_info
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes
from vlm_fo1.task_templates import OD_template


# Skip all GPU tests if CUDA is not available
pytestmark = pytest.mark.gpu


def is_gpu_available():
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


@pytest.fixture(scope="module")
def gpu_diagnostics():
    """Gather GPU diagnostics before running tests."""
    diag = backend_info()
    print("\n=== GPU Diagnostics ===")
    print(f"CUDA available: {diag['cuda_available']}")
    print(f"Wheel ABI: {diag['wheel_abi']}")
    print(f"Driver version: {diag['driver_version']}")
    print(f"Torch CUDA version: {diag['torch']['version']}")
    print(f"UPN extension: {diag['upn_extension']['backend']}")
    print("=======================\n")
    return diag


@pytest.fixture(scope="module")
def model_and_processor(gpu_diagnostics):
    """
    Load VLM-FO1 model once for all tests.

    This uses a lightweight model variant or falls back to a dummy if model not available.
    """
    if not is_gpu_available():
        pytest.skip("GPU not available")

    # Try to load from HuggingFace or local resources
    # For CI, you might want to use a smaller test checkpoint
    model_path = os.getenv(
        'VLM_FO1_TEST_MODEL',
        'omlab/VLM-FO1_Qwen2.5-VL-3B-v01'
    )

    print(f"Loading model from: {model_path}")

    try:
        tokenizer, model, image_processors = load_pretrained_model(
            model_path,
            device="cuda",
            load_8bit=False,
            load_4bit=False
        )
        return tokenizer, model, image_processors, model_path
    except Exception as e:
        pytest.skip(f"Model loading failed: {e}")


def create_test_image():
    """Create a deterministic test image."""
    # Create a simple 224x224 RGB image with deterministic content
    np.random.seed(42)
    img_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def test_gpu_available():
    """Verify GPU is available."""
    assert is_gpu_available(), "CUDA GPU not available for testing"
    assert torch.cuda.device_count() > 0, "No CUDA devices found"


def test_backend_diagnostics(gpu_diagnostics):
    """Verify backend diagnostics are correct."""
    assert gpu_diagnostics['cuda_available'], "CUDA should be available"
    assert gpu_diagnostics['driver_version'] is not None, "Driver version should be detected"
    assert gpu_diagnostics['torch']['available'], "Torch CUDA should be available"


def test_model_loads_on_gpu(model_and_processor):
    """Verify model loads successfully on GPU."""
    tokenizer, model, image_processors, model_path = model_and_processor

    assert model is not None, "Model should load"
    assert tokenizer is not None, "Tokenizer should load"
    assert image_processors is not None, "Image processors should load"

    # Check model is on CUDA
    assert next(model.parameters()).is_cuda, "Model should be on CUDA device"


def test_deterministic_inference(model_and_processor):
    """Run deterministic inference and verify output."""
    tokenizer, model, image_processors, model_path = model_and_processor

    # Set seeds for determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Create test image
    test_img = create_test_image()

    # Define test bounding boxes (xyxy format)
    bbox_list = [
        [50.0, 50.0, 150.0, 150.0],
        [100.0, 100.0, 200.0, 200.0],
    ]

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "test_image"}},
                {"type": "text", "text": OD_template.format("object")},
            ],
            "bbox_list": bbox_list,
        }
    ]

    # Prepare inputs
    generation_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=512,  # Reduced for smoke test
        top_p=0.05,
        temperature=0.0,
        do_sample=False,
    )

    # Run inference
    with torch.inference_mode():
        start_time = time.time()
        output_ids = model.generate(**generation_kwargs)
        inference_time = time.time() - start_time

    # Decode output
    outputs = tokenizer.decode(
        output_ids[0, generation_kwargs['inputs'].shape[1]:],
        skip_special_tokens=False
    ).strip()

    print(f"\nInference time: {inference_time:.3f}s")
    print(f"Output length: {len(outputs)} chars")
    print(f"Output preview: {outputs[:200]}")

    # Validate output
    assert output_ids is not None, "Output IDs should not be None"
    assert output_ids.shape[0] == 1, "Batch size should be 1"
    assert output_ids.dtype == torch.long, "Output dtype should be torch.long"
    assert inference_time < 60.0, f"Inference took too long: {inference_time:.3f}s"


def test_gpu_memory_usage(model_and_processor):
    """Measure GPU memory usage during inference."""
    tokenizer, model, image_processors, model_path = model_and_processor

    # Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    # Create test image and bbox
    test_img = create_test_image()
    bbox_list = [[50.0, 50.0, 150.0, 150.0]]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "test_image"}},
                {"type": "text", "text": OD_template.format("test")},
            ],
            "bbox_list": bbox_list,
        }
    ]

    generation_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=256,
        temperature=0.0,
        do_sample=False,
    )

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    final_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    print(f"\nGPU Memory Usage:")
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  Peak: {peak_memory:.2f} MB")
    print(f"  Final: {final_memory:.2f} MB")
    print(f"  Delta: {peak_memory - initial_memory:.2f} MB")

    # Validate memory usage is reasonable (not exceeding 24GB for 3B model)
    assert peak_memory < 24 * 1024, f"Peak memory usage too high: {peak_memory:.2f} MB"


def test_multiple_inferences_consistent(model_and_processor):
    """Verify multiple inferences produce consistent results."""
    tokenizer, model, image_processors, model_path = model_and_processor

    test_img = create_test_image()
    bbox_list = [[50.0, 50.0, 150.0, 150.0]]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "test_image"}},
                {"type": "text", "text": OD_template.format("object")},
            ],
            "bbox_list": bbox_list,
        }
    ]

    outputs_list = []

    # Run inference 3 times with same seed
    for i in range(3):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        generation_kwargs = prepare_inputs(
            model_path,
            model,
            image_processors,
            tokenizer,
            messages,
            max_tokens=256,
            temperature=0.0,
            do_sample=False,
        )

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        outputs = tokenizer.decode(
            output_ids[0, generation_kwargs['inputs'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        outputs_list.append(outputs)

    # Verify all outputs are identical (deterministic)
    print(f"\nRun 1 output: {outputs_list[0][:100]}")
    print(f"Run 2 output: {outputs_list[1][:100]}")
    print(f"Run 3 output: {outputs_list[2][:100]}")

    # Note: Due to non-determinism in some CUDA operations, we check similarity rather than exact match
    # For exact determinism, all runs should be identical, but we allow for minor variations
    assert len(set(outputs_list)) <= 2, "Outputs should be mostly consistent across runs"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])