"""
VLM-FO1 inference on Modal AI with unified dependencies.

This example shows how to deploy VLM-FO1 on Modal with guaranteed dependency compatibility.

Usage:
    modal deploy examples/modal_app.py
    modal run examples/modal_app.py::run_inference --image-path "path/to/image.jpg"
"""

import modal

# Create Modal app
app = modal.App("vlm-fo1-inference")

# Define image with exact dependencies from pyproject.toml
# This ensures compatibility across all platforms
vlm_fo1_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.10"
    )
    # Install torch with CUDA 11.8 (matches our wheel ABI)
    .pip_install(
        "torch>=2.1,<2.7",
        "torchvision>=0.16",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    # Install VLM-FO1 dependencies (matches pyproject.toml)
    .pip_install(
        "transformers>=4.45,<5.0",
        "timm>=0.9.0",
        "accelerate>=1.0",
        "safetensors>=0.4.0",
        "pillow>=9.0",
        "numpy>=1.21",
    )
    # Install VLM-FO1 from source (or use wheel URL from GitHub releases)
    .pip_install("git+https://github.com/om-ai-lab/VLM-FO1.git")
    # Or from local directory:
    # .copy_local_dir(".", "/vlm-fo1")
    # .run_commands("cd /vlm-fo1 && BUILD_CUDA=cu118 pip install -e .")
)

# Mount for storing models (persists across runs)
model_volume = modal.Volume.from_name("vlm-fo1-models", create_if_missing=True)


@app.function(
    image=vlm_fo1_image,
    gpu=modal.gpu.T4(),  # or A10G, A100
    timeout=600,
    volumes={"/models": model_volume},
)
def run_inference(image_path: str, query: str = "orange", use_upn: bool = False):
    """
    Run VLM-FO1 inference on Modal.

    Args:
        image_path: Path to image file
        query: Object to detect
        use_upn: Whether to use UPN detector for proposals

    Returns:
        dict with predictions and bounding boxes
    """
    import torch
    from PIL import Image
    from vlm_fo1.model.builder import load_pretrained_model
    from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes
    from vlm_fo1.task_templates import OD_template

    # Verify environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model (cached in volume)
    model_path = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
    print(f"Loading model: {model_path}")

    tokenizer, model, image_processors = load_pretrained_model(
        model_path,
        device="cuda",
        load_8bit=False,
    )

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Generate proposals (simplified - you can add UPN here)
    bbox_list = [[50, 50, 150, 150], [100, 100, 200, 200]]

    # Prepare messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": OD_template.format(query)},
        ],
        "bbox_list": bbox_list,
    }]

    # Run inference
    generation_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=512,
        temperature=0.0,
        do_sample=False,
    )

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    outputs = tokenizer.decode(
        output_ids[0, generation_kwargs['inputs'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Extract bounding boxes
    bboxes = extract_predictions_to_bboxes(outputs, bbox_list)

    return {
        "query": query,
        "output": outputs,
        "bboxes": bboxes,
        "image_size": img.size,
    }


@app.local_entrypoint()
def main(image_path: str = "demo/demo_image.jpg", query: str = "orange"):
    """Run inference from local machine."""
    result = run_inference.remote(image_path, query)
    print(f"\nResults for query '{query}':")
    print(f"Output: {result['output']}")
    print(f"Bounding boxes: {result['bboxes']}")


if __name__ == "__main__":
    # For testing locally with modal run
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "orange")
