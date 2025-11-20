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
        "einops",
        "scikit-image>=0.20.0",
        "mmengine==0.8.2",  # Required for UPN
        "mmcv>=2.0.0"  # Required for UPN
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
        from detect_tools.upn import UPNWrapper
        
        # Check if model exists in volume, if not download it
        # Note: load_pretrained_model usually handles downloading, but we want to ensure
        # it uses the cache directory /models
        os.environ["HF_HOME"] = "/models"
        
        print(f"Loading VLM-FO1 model: {MODEL_PATH}")
        # Explicitly use sdpa to avoid flash-attn build issues
        self.tokenizer, self.model, self.image_processors = load_pretrained_model(
            MODEL_PATH, 
            device="cuda", 
            attn_implementation="sdpa"
        )
        print("✓ VLM-FO1 Model loaded successfully!")
        
        # Initialize UPN for region proposals
        print("Loading UPN (Universal Proposal Network)...")
        # UPN checkpoint should be in the volume or we download it
        upn_path = "/models/upn_large.pth"
        if not os.path.exists(upn_path):
            print("Downloading UPN checkpoint...")
            import requests
            upn_url = "https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth"
            response = requests.get(upn_url, stream=True)
            response.raise_for_status()
            os.makedirs(os.path.dirname(upn_path), exist_ok=True)
            with open(upn_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✓ UPN checkpoint downloaded")
        
        self.upn_model = UPNWrapper(upn_path)
        print("✓ UPN loaded successfully!")
        print("=" * 60)
        print("Container ready for inference!")
        print("=" * 60)

    @modal.method()
    def generate(self, image_url: str, prompt: str = "orange", threshold: float = 0.3):
        """Runs inference on the provided image and prompt."""
        import torch
        from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_indexes
        from vlm_fo1.task_templates import OD_template
        import requests
        from PIL import Image
        from io import BytesIO

        print(f"Processing request for: '{prompt}'")
        
        # Load Image
        if image_url.startswith("http"):
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_url).convert("RGB")

        # Step 1: Get region proposals from UPN
        print(f"Detecting regions with UPN (threshold={threshold})...")
        proposals = self.upn_model.inference(img)
        filtered_proposals = self.upn_model.filter(proposals, min_score=threshold)
        bbox_list = filtered_proposals['original_xyxy_boxes'][0][:100]  # Limit to 100 boxes
        print(f"✓ Found {len(bbox_list)} region proposals")

        # Save to tempfile for the model processor
        temp_path = "/tmp/input_image.jpg"
        img.save(temp_path)

        # Step 2: Prepare Input for VLM-FO1
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": temp_path},
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

        # Step 3: Run VLM-FO1 to identify which regions match
        print("Running VLM-FO1 inference...")
        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)
            outputs = self.tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
        
        print(f"Raw output: {outputs}")

        # Step 4: Extract bounding boxes from the prediction
        prediction_dict = extract_predictions_to_indexes(outputs)
        
        detected_objects = []
        
        for label, box_indexes in prediction_dict.items():
            for box_index in box_indexes:
                box_index = int(box_index)
                if box_index < len(bbox_list):
                    current_bbox = bbox_list[box_index]
                    detected_objects.append({
                        "bbox": [float(current_bbox[0]), float(current_bbox[1]), 
                                float(current_bbox[2]), float(current_bbox[3])],
                        "label": label,
                        "region_id": box_index,
                        "confidence": 1.0  # VLM-FO1 is binary (selected or not)
                    })
        
        print(f"✓ Detected {len(detected_objects)} objects matching '{prompt}'")

        return {
            "raw_output": outputs,
            "prompt": prompt,
            "detected": len(detected_objects) > 0,
            "num_detections": len(detected_objects),
            "detections": detected_objects,  # List of {bbox: [x1,y1,x2,y2], label: str, region_id: int}
            "num_proposals": len(bbox_list),
            "image_size": {"width": img.size[0], "height": img.size[1]}
        }

    @modal.fastapi_endpoint(method="POST")
    def web_generate(self, item: dict):
        """Exposes the generation logic as a REST API."""
        # item expects {"image_url": "...", "prompt": "...", "threshold": 0.3}
        return self.generate.local(
            item.get("image_url"), 
            item.get("prompt", "orange"),
            item.get("threshold", 0.3)
        )

@app.local_entrypoint()
def main():
    print("Triggering remote inference...")
    model = VLMInference()
    
    # Test with the demo image mounted in the container
    # Note: In remote execution, the path is /root/demo/demo_image.jpg
    result = model.generate.remote("/root/demo/demo_image.jpg", "orange")
    
    print(f"Final Result received locally: {result}")
