import modal
from modal import App, Image, Volume
import os
import sys
from pathlib import Path

# Define the image with all dependencies
gradio_image = (
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
        "einops",
        "scikit-image>=0.20.0",
        "gradio>=4.0.0",  # Gradio for web UI
        "mmengine==0.8.2",  # Required for UPN
        "mmcv>=2.0.0"  # Required for UPN
    )
    .add_local_dir("vlm_fo1", remote_path="/root/vlm_fo1")
    .add_local_dir("detect_tools", remote_path="/root/detect_tools")
    .add_local_dir("demo", remote_path="/root/demo")
)

app = modal.App("vlm-fo1-gradio")

# Model cache volume
MODEL_PATH = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
MODEL_VOL = Volume.from_name("vlm-fo1-cache", create_if_missing=True)
VOLUMES = {Path("/models"): MODEL_VOL}

@app.function(
    gpu="A100",
    image=gradio_image,
    volumes=VOLUMES,
    timeout=1800,  # 30 minutes
)
@modal.asgi_app()
def gradio_app():
    """Gradio web interface for VLM-FO1"""
    import gradio as gr
    from PIL import Image, ImageDraw
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, disk
    import torch
    import re
    
    sys.path.append("/root")
    
    from detect_tools.upn import UPNWrapper
    from vlm_fo1.model.builder import load_pretrained_model
    from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_indexes
    from vlm_fo1.task_templates import OD_template, OD_Counting_template, Grounding_template
    
    # Set environment
    os.environ["HF_HOME"] = "/models"
    
    print("Loading VLM-FO1 model...")
    tokenizer, model, image_processors = load_pretrained_model(
        MODEL_PATH,
        device="cuda",
        attn_implementation="sdpa"
    )
    print("✓ Model loaded!")
    
    print("Loading UPN...")
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
    
    upn_model = UPNWrapper(upn_path)
    print("✓ UPN loaded!")
    
    # Task templates
    TASK_TYPES = {
        "Object Detection": OD_template,
        "Object Counting": OD_Counting_template,
        "Grounding": Grounding_template,
    }
    
    def detect_regions(image, threshold=0.3):
        """Use UPN to detect regions"""
        proposals = upn_model.inference(image)
        filtered_proposals = upn_model.filter(proposals, min_score=threshold)
        return filtered_proposals['original_xyxy_boxes'][0][:100]
    
    def draw_bboxes(image, bboxes):
        """Draw bounding boxes on image"""
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(bbox, outline="red", width=3)
        return image
    
    def process_image(image, prompt, task_type, threshold):
        """Main processing function"""
        if image is None:
            return None, None, "Please upload an image", []
        
        image_rgb = image.convert('RGB')
        
        # Get region proposals
        bbox_list = detect_regions(image_rgb, threshold)
        
        # Prepare template
        template = TASK_TYPES[task_type]
        text = template.format(prompt)
        
        # Save temp
        temp_path = "/tmp/gradio_input.jpg"
        image_rgb.save(temp_path)
        
        # Prepare messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": temp_path}},
                {"type": "text", "text": text}
            ],
            "bbox_list": bbox_list,
        }]
        
        generation_kwargs = prepare_inputs(
            MODEL_PATH, model, image_processors, tokenizer, messages,
            max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
        )
        
        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)
            outputs = tokenizer.decode(
                output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
        
        # Extract predictions
        prediction_dict = extract_predictions_to_indexes(outputs)
        
        # Build detection results
        detected_bboxes = []
        annotated_bboxes = []
        
        for label, box_indexes in prediction_dict.items():
            for box_index in box_indexes:
                box_index = int(box_index)
                if box_index < len(bbox_list):
                    bbox = bbox_list[box_index]
                    detected_bboxes.append({
                        "label": label,
                        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        "region_id": box_index
                    })
                    annotated_bboxes.append(
                        ((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), label)
                    )
        
        # Draw proposals
        image_with_proposals = draw_bboxes(image_rgb, bbox_list)
        
        # Create annotated image
        annotated_image = (image_rgb, annotated_bboxes)
        
        return annotated_image, image_with_proposals, outputs, detected_bboxes
    
    # Build Gradio interface
    demo = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(label="Prompt", placeholder="e.g., person, car, dog", value="person"),
            gr.Dropdown(choices=list(TASK_TYPES.keys()), value="Object Detection", label="Task Type"),
            gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.1, label="Detection Threshold")
        ],
        outputs=[
            gr.AnnotatedImage(label="Detected Objects"),
            gr.Image(label="All Proposals (UPN)"),
            gr.Textbox(label="Model Output", lines=3),
            gr.JSON(label="Detection Results")
        ],
        title="🔍 VLM-FO1: Vision-Language Model with Fine-grained Object Understanding",
        description="""
        Upload an image and enter a prompt to detect objects, count them, or perform grounding.
        The model uses UPN to generate region proposals, then VLM-FO1 identifies matching regions.
        
        **First request will take 30-60s** (model loading + UPN download), then it's fast!
        """,
        allow_flagging="never"
    )
    
    return demo
