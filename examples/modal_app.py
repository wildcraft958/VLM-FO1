"""
VLM-FO1 inference on Modal AI with unified dependencies and SAM3 support.

This example shows how to deploy VLM-FO1 on Modal with guaranteed dependency compatibility.
Supports both UPN and SAM3 as bbox proposal generators.

Usage:
    modal deploy examples/modal_app.py
    modal run examples/modal_app.py::run_inference --image-path "path/to/image.jpg" --query "orange"
    modal run examples/modal_app.py::run_inference_with_sam3 --image-path "path/to/image.jpg" --query "the ball nearest to the bear"
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
    .apt_install("git", "g++", "libgl1-mesa-glx", "libglib2.0-0")
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
        "fastapi",  # Required for web endpoints
        "uvicorn[standard]",  # ASGI server for FastAPI
        "python-multipart",  # For FastAPI form data
    )
    # Install SAM3 for bbox proposals (optional, but recommended)
    # Note: SAM3 requires access to checkpoints from HuggingFace
    # Install SAM3 dependencies first
    .pip_install(
        "einops",
        "decord",
        "pycocotools",
        "opencv-python",  # SAM3 dependency
        "scikit-image",  # SAM3 dependency for visualization
        "huggingface-hub",  # For downloading SAM3 checkpoints
        "matplotlib",  # SAM3 visualization dependency
    )
    # Install SAM3 by cloning and installing in editable mode
    # This ensures all submodules and internal packages are properly included
    .run_commands(
        "cd /tmp && "
        "git clone --depth 1 --recursive https://github.com/facebookresearch/sam3.git sam3_repo && "
        "cd sam3_repo && "
        "pip install -e ."
    )
    # Copy local VLM-FO1 codebase (don't install, just add to path like modal_inference.py)
    # This avoids setuptools-scm version detection issues
    .add_local_dir("vlm_fo1", remote_path="/root/vlm_fo1")
    .add_local_dir("detect_tools", remote_path="/root/detect_tools")
)

# Mount for storing models (persists across runs)
model_volume = modal.Volume.from_name("vlm-fo1-models", create_if_missing=True)

# HuggingFace secret for accessing gated repositories (e.g., SAM3)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=vlm_fo1_image,
    gpu="A10G",  # or "T4", "A100"
    timeout=600,
    volumes={"/models": model_volume},
)
def run_inference(image_path: str, query: str = "orange", use_upn: bool = False):
    """
    Run VLM-FO1 inference on Modal with manual bbox proposals.

    Args:
        image_path: Path to image file
        query: Object to detect
        use_upn: Whether to use UPN detector for proposals (not implemented in this function)

    Returns:
        dict with predictions and bounding boxes
    """
    import sys
    import torch
    from PIL import Image
    
    # Add /root to path so we can import vlm_fo1 (like modal_inference.py)
    sys.path.append("/root")
    
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


@app.function(
    image=vlm_fo1_image,
    gpu="A10G",  # or "T4", "A100"
    timeout=600,
    volumes={"/models": model_volume},
    secrets=[hf_secret],  # Add HF secret for SAM3 access
)
def run_inference_with_sam3(
    image_path: str = None,
    image_bytes: bytes = None,
    query: str = "the ball nearest to the bear",
    sam3_model_path: str = "/models/sam3/sam3.pt",
    confidence_threshold: float = 0.5,
    max_proposals: int = 100,
):
    """
    Run VLM-FO1 inference on Modal with SAM3 as bbox proposal generator.

    Args:
        image_path: Path to image file or URL (optional if image_bytes provided)
        image_bytes: Raw image bytes (optional if image_path provided)
        query: Text prompt describing the object to detect (e.g., "the ball nearest to the bear")
        sam3_model_path: Path to SAM3 checkpoint (default: /models/sam3/sam3.pt)
        confidence_threshold: Confidence threshold for SAM3 proposals (default: 0.5)
        max_proposals: Maximum number of proposals to use (default: 100)

    Returns:
        dict with predictions, bounding boxes, and SAM3 proposals
    """
    import sys
    import torch
    from PIL import Image
    import os
    
    # Add /root to path so we can import vlm_fo1 (like modal_inference.py)
    sys.path.append("/root")
    
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from vlm_fo1.model.builder import load_pretrained_model
    from vlm_fo1.mm_utils import (
        prepare_inputs,
        extract_predictions_to_indexes,
        extract_predictions_to_bboxes,
    )
    from vlm_fo1.task_templates import OD_template

    # Verify environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set HuggingFace token from secret for accessing gated repositories
    # The secret should have HF_TOKEN or HUGGINGFACE_HUB_TOKEN key
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        # Login to HuggingFace Hub to authenticate
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("✓ HuggingFace token found and authenticated for gated repositories")
        except Exception as e:
            print(f"⚠ Warning: Failed to login to HuggingFace: {e}")
            print("Continuing anyway - token may still work...")
    else:
        print("⚠ Warning: No HuggingFace token found. SAM3 may fail if it requires authentication.")
        print("Make sure your Modal secret 'huggingface-secret' has 'HF_TOKEN' or 'HUGGINGFACE_HUB_TOKEN' key.")
    
    # Load SAM3 model
    print(f"Loading SAM3 model from: {sam3_model_path}")
    if not os.path.exists(sam3_model_path):
        # Try to download or use default HuggingFace model
        print(f"SAM3 checkpoint not found at {sam3_model_path}, trying to load from HuggingFace...")
        try:
            sam3_model = build_sam3_image_model(device="cuda")
        except Exception as e:
            print(f"❌ Error loading SAM3 from HuggingFace: {e}")
            print("Make sure you have access to the SAM3 repository and your HF token is set correctly.")
            raise
    else:
        sam3_model = build_sam3_image_model(checkpoint_path=sam3_model_path, device="cuda")
    
    sam3_processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold, device="cuda")
    print("SAM3 model loaded successfully")

    # Load VLM-FO1 model
    model_path = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
    print(f"Loading VLM-FO1 model: {model_path}")
    tokenizer, model, image_processors = load_pretrained_model(
        model_path,
        device="cuda",
        load_8bit=False,
    )
    print("VLM-FO1 model loaded successfully")

    # Load and preprocess image
    # Support both file path and image bytes - prioritize bytes for web endpoints
    from io import BytesIO
    
    if image_bytes is not None:
        # Use image bytes directly (preferred for web endpoints)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        print(f"Image loaded from bytes: {img.size}")
    elif image_path and image_path.startswith("http"):
        # Download from URL
        import requests
        print(f"Downloading image from URL: {image_path}")
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Image downloaded and loaded: {img.size}")
    elif image_path and os.path.exists(image_path):
        # Load from file path
        img = Image.open(image_path).convert("RGB")
        print(f"Image loaded from file: {img.size}")
    else:
        raise ValueError(
            f"Image not found. image_path={image_path}, has_bytes={image_bytes is not None}. "
            "Provide either image_path (existing file or URL) or image_bytes."
        )
    print(f"Image loaded: {img.size}")

    # Run SAM3 to get fine-grained object proposals
    print(f"Running SAM3 with query: '{query}'")
    inference_state = sam3_processor.set_image(img)
    sam3_processor.reset_all_prompts(inference_state)
    output = sam3_processor.set_text_prompt(state=inference_state, prompt=query)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"SAM3 found {len(boxes)} proposals")

    # Sort by scores from high to low and take top proposals
    sorted_indices = torch.argsort(scores, descending=True)
    masks = masks[sorted_indices][:max_proposals]
    boxes = boxes[sorted_indices][:max_proposals]
    scores = scores[sorted_indices][:max_proposals]

    print(f"Using top {len(boxes)} proposals (scores: {scores[:5].tolist()})")

    # Prepare chat messages with vision input and bounding boxes
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_path},
                },
                {
                    "type": "text",
                    "text": OD_template.format(query),
                },
            ],
            "bbox_list": boxes.tolist(),
        }
    ]

    # Run VLM-FO1 inference
    print("Running VLM-FO1 inference...")
    generation_kwargs = prepare_inputs(
        model_path,
        model,
        image_processors,
        tokenizer,
        messages,
        max_tokens=4096,
        top_p=0.05,
        temperature=0.0,
        do_sample=False,
    )

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    outputs = tokenizer.decode(
        output_ids[0, generation_kwargs['inputs'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"VLM-FO1 output: {outputs}")

    # Convert output prediction (indexes) to bounding box coordinates
    bbox_indexes = extract_predictions_to_indexes(outputs)
    
    # Map indexes to actual bounding boxes
    res = {}
    res_masks = []
    for label, index_set in bbox_indexes.items():
        if label not in res:
            res[label] = []
        for i in index_set:
            if i < len(boxes):
                res[label].append(boxes[i].tolist())
                res_masks.append(masks[i].tolist())

    return {
        "query": query,
        "output": outputs,
        "bboxes": res,
        "sam3_proposals": {
            "count": len(boxes),
            "top_scores": scores[:10].tolist(),
        },
        "image_size": img.size,
    }


# Web endpoints for frontend integration
# Using asgi_app for full FastAPI control with CORS support
@app.function(
    image=vlm_fo1_image,
    gpu="A10G",
    timeout=600,
    volumes={"/models": model_volume},
    secrets=[hf_secret],  # Add HF secret for SAM3 access in web endpoints
)
@modal.asgi_app()
def web_api():
    """FastAPI app with CORS support for web endpoints."""
    from fastapi import FastAPI, Response, Request
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    import base64
    from io import BytesIO
    from PIL import Image
    import tempfile
    import os
    import json
    
    app = FastAPI(title="VLM-FO1 Inference API")
    
    # Add comprehensive CORS middleware to handle all frontend requests
    # This allows requests from any origin (for development)
    # In production, replace ["*"] with specific allowed origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=False,  # Must be False when using allow_origins=["*"]
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],  # Explicitly allow all methods
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ],
        expose_headers=["*"],  # Expose all headers to the frontend
        max_age=3600,  # Cache preflight requests for 1 hour
    )
    
    # Add a backup middleware to ensure CORS headers are always present on all responses
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class CORSBackupMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            try:
                response = await call_next(request)
            except Exception as e:
                # Even on errors, return CORS headers
                from starlette.responses import JSONResponse
                import traceback
                response = JSONResponse(
                    content={
                        "error": str(e),
                        "type": type(e).__name__,
                    },
                    status_code=500
                )
            # Add CORS headers to all responses (including errors) - CRITICAL
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Origin, Accept"
            response.headers["Access-Control-Expose-Headers"] = "*"
            return response
    
    app.add_middleware(CORSBackupMiddleware)
    
    # Add explicit OPTIONS handlers for CORS preflight (middleware should handle this, but explicit is safer)
    @app.options("/web_inference")
    async def options_web_inference():
        """Handle CORS preflight requests for web_inference."""
        return Response(
            content="",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Origin",
                "Access-Control-Max-Age": "3600",
            }
        )
    
    @app.options("/web_inference_sam3")
    async def options_web_inference_sam3():
        """Handle CORS preflight requests for web_inference_sam3."""
        return Response(
            content="",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Origin",
                "Access-Control-Max-Age": "3600",
            }
        )
    
    # Add a health check endpoint
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "message": "VLM-FO1 Inference API is running"}
    
    @app.post("/web_inference")
    async def web_inference(item: dict):
        """
        Web endpoint for VLM-FO1 inference with manual bbox proposals.
        
        Expects: {"image_url": "...", "query": "..."}
        Returns: {"output": "...", "bboxes": {...}, "image_size": [...]}
        """
        image_url = item.get("image_url", "")
        query = item.get("query", "orange")
        
        # Handle base64 data URLs or regular URLs
        if image_url.startswith("data:image"):
            base64_data = image_url.split(",", 1)[1]
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data)).convert("RGB")
            # Save to temp file for the inference function
            temp_path = "/tmp/input_image.jpg"
            img.save(temp_path)
            image_path = temp_path
        elif image_url.startswith("http"):
            # For URLs, pass directly
            image_path = image_url
        else:
            image_path = image_url
        
        result = run_inference.remote(image_path, query)
        
        # Format response for frontend
        detections = []
        for label, bbox_list in result.get("bboxes", {}).items():
            for bbox in bbox_list:
                detections.append({
                    "bbox": bbox,
                    "label": label,
                })
        
        # Return response with explicit CORS headers (backup to middleware)
        response_data = {
            "raw_output": result.get("output", ""),
            "prompt": query,
            "detected": len(detections) > 0,
            "num_detections": len(detections),
            "detections": detections,
            "image_size": result.get("image_size", {}),
        }
        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            }
        )
    
    @app.post("/web_inference_sam3")
    async def web_inference_sam3(item: dict):
        """
        Web endpoint for VLM-FO1 inference with SAM3 bbox proposals.
        
        Expects: {"image_url": "...", "query": "...", "confidence_threshold": 0.5, "max_proposals": 100}
        Returns: {"output": "...", "bboxes": {...}, "sam3_proposals": {...}, "image_size": [...]}
        """
        try:
            # Ensure HF token is set for SAM3 access (secret should provide this via environment)
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            
            image_url = item.get("image_url", "")
            query = item.get("query", "the ball nearest to the bear")
            confidence_threshold = item.get("confidence_threshold", 0.5)
            max_proposals = item.get("max_proposals", 100)
            sam3_model_path = item.get("sam3_model_path", "/models/sam3/sam3.pt")
            
            # Handle base64 data URLs or regular URLs
            # Pass image bytes directly to avoid file system issues across Modal function calls
            image_bytes = None
            image_path_for_remote = None
            
            if image_url.startswith("data:image"):
                base64_data = image_url.split(",", 1)[1]
                image_bytes = base64.b64decode(base64_data)
                # Don't save to file - pass bytes directly
            elif image_url.startswith("http"):
                # For URLs, pass the URL directly
                image_path_for_remote = image_url
            else:
                # Assume it's a file path (unlikely in web context, but handle it)
                image_path_for_remote = image_url
            
            # Call remote function with image bytes or path
            if image_bytes is not None:
                result = run_inference_with_sam3.remote(
                    image_bytes=image_bytes,
                    query=query,
                    sam3_model_path=sam3_model_path,
                    confidence_threshold=confidence_threshold,
                    max_proposals=max_proposals,
                )
            else:
                result = run_inference_with_sam3.remote(
                    image_path=image_path_for_remote,
                    query=query,
                    sam3_model_path=sam3_model_path,
                    confidence_threshold=confidence_threshold,
                    max_proposals=max_proposals,
                )
            
            # Format response for frontend
            detections = []
            for label, bbox_list in result.get("bboxes", {}).items():
                for bbox in bbox_list:
                    detections.append({
                        "bbox": bbox,
                        "label": label,
                    })
            
            # Return response with explicit CORS headers (backup to middleware)
            response_data = {
                "raw_output": result.get("output", ""),
                "prompt": query,
                "detected": len(detections) > 0,
                "num_detections": len(detections),
                "detections": detections,
                "sam3_proposals": result.get("sam3_proposals", {}),
                "image_size": result.get("image_size", {}),
            }
            return Response(
                content=json.dumps(response_data),
                media_type="application/json",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
                }
            )
        except Exception as e:
            # Return error with CORS headers
            import traceback
            error_response = {
                "error": str(e),
                "type": type(e).__name__,
            }
            return Response(
                content=json.dumps(error_response),
                media_type="application/json",
                status_code=500,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
                }
            )
    
    return app


@app.local_entrypoint()
def main(
    image_path: str = "demo/demo_image.jpg",
    query: str = "orange",
    use_sam3: bool = False,
    sam3_model_path: str = "/models/sam3/sam3.pt",
):
    """
    Run inference from local machine.
    
    Args:
        image_path: Path to image file
        query: Object to detect or text prompt
        use_sam3: Whether to use SAM3 for bbox proposals
        sam3_model_path: Path to SAM3 checkpoint (if using SAM3)
    """
    if use_sam3:
        print(f"Running inference with SAM3...")
        result = run_inference_with_sam3.remote(
            image_path,
            query,
            sam3_model_path=sam3_model_path,
        )
    else:
        print(f"Running inference with manual bbox proposals...")
        result = run_inference.remote(image_path, query)
    
    print(f"\nResults for query '{query}':")
    print(f"Output: {result['output']}")
    print(f"Bounding boxes: {result['bboxes']}")
    if use_sam3 and 'sam3_proposals' in result:
        print(f"SAM3 proposals: {result['sam3_proposals']}")


if __name__ == "__main__":
    # For testing locally with modal run
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "orange")
