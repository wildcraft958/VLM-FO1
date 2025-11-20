import modal
from modal import App, Image, Volume
import os
import sys
import time
from pathlib import Path

# 1. Define Image with FlashAttention2 and UV
# We use a specific CUDA-compatible image and PIN versions to ensure we get pre-built wheels
# instead of compiling from source (which takes forever on CPU-only builders).
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "g++", "libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers>=4.45",
        "timm>=0.9.0",
        "accelerate>=1.0",
        "safetensors>=0.4.0",
        "pillow>=9.0",
        "numpy<2.0",
        "ninja",
        "packaging",
        "wheel",
        "huggingface_hub",
        "fastapi",
        "pydantic",
        "einops"
    )
    .add_local_dir("vlm_fo1", remote_path="/root/vlm_fo1")
    .add_local_dir("detect_tools", remote_path="/root/detect_tools")
    .add_local_dir("demo", remote_path="/root/demo")
)

app = modal.App("vlm-fo1-inference")

# 2. Model Volume for Caching
# This volume will persist across runs, saving download time
MODEL_PATH = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
MODEL_VOL = Volume.from_name("vlm-fo1-cache", create_if_missing=True)
VOLUMES = {Path("/models"): MODEL_VOL}

@app.cls(
    gpu="A100",  # Explicitly request A100 for FlashAttention2 support
    image=image,
    volumes=VOLUMES,
    timeout=600, # 10 minutes timeout
    scaledown_window=300 # Keep container alive for 5 mins after request
)
class VLMInference:
    @modal.enter()
    def load_model(self):
        """Loads the model once when the container starts."""
        print("Initializing VLM-FO1 Inference Container...")
        import torch
        
        # Add /root to path so we can import vlm_fo1
        sys.path.append("/root")
        
        from vlm_fo1.model.builder import load_pretrained_model
        
        # Check if model exists in volume, if not download it
        # Note: load_pretrained_model usually handles downloading, but we want to ensure
        # it uses the cache directory /models
        os.environ["HF_HOME"] = "/models"
        
        print(f"Loading model: {MODEL_PATH}")
        # Explicitly use sdpa to avoid flash-attn build issues
        self.tokenizer, self.model, self.image_processors = load_pretrained_model(
            MODEL_PATH, 
            device="cuda", 
            attn_implementation="sdpa"
        )
        print("Model loaded successfully with SDPA (FlashAttention disabled)!")

    @modal.method()
    def generate(self, image_url: str, prompt: str = "orange"):
        """Runs inference on the provided image and prompt."""
        import torch
        from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes
        from vlm_fo1.task_templates import OD_template
        import requests
        from PIL import Image
        from io import BytesIO

        print(f"Processing request for: {prompt}")
        
        # Load Image
        if image_url.startswith("http"):
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            # Save to temp file for the model processor (which expects a path or url structure)
            # For simplicity in this demo, we'll save it locally in the container
            temp_path = "/tmp/input_image.jpg"
            img.save(temp_path)
            img_path = temp_path
        else:
            # Assume it's a local path in the container (e.g. from mount)
            img_path = image_url

        # Prepare Input
        # Dummy bbox list for OD task
        bbox_list = [] 

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_path},
                    },
                    {
                        "type": "text",
                        "text": OD_template.format(prompt),
                    },
                ],
                "bbox_list": bbox_list,
            }
        ]

        generation_kwargs = prepare_inputs(
            MODEL_PATH, self.model, self.image_processors, self.tokenizer, messages,
            max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
        )

        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)
            outputs = self.tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()

        return outputs

    @modal.fastapi_endpoint(method="POST")
    def web_generate(self, item: dict):
        """Exposes the generation logic as a REST API."""
        # item expects {"image_url": "...", "prompt": "..."}
        return {"result": self.generate.local(item.get("image_url"), item.get("prompt", "orange"))}

@app.local_entrypoint()
def main():
    print("Triggering remote inference...")
    model = VLMInference()
    
    # Test with the demo image mounted in the container
    # Note: In remote execution, the path is /root/demo/demo_image.jpg
    result = model.generate.remote("/root/demo/demo_image.jpg", "orange")
    
    print(f"Final Result received locally: {result}")
