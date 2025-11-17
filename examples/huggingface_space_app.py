"""
VLM-FO1 Hugging Face Spaces App with unified dependencies.

This example shows how to deploy VLM-FO1 on Hugging Face Spaces
with guaranteed dependency compatibility.

File structure for your Space:
    app.py                  (this file)
    requirements.txt        (see below)
    README.md              (optional)

Space Configuration:
    - SDK: Gradio
    - Python version: 3.10
    - Hardware: GPU T4 (or better)
"""

import gradio as gr
import os
import sys

# Ensure compatibility
os.environ['BUILD_CUDA'] = 'cu118'

# Import after environment setup
import torch
from PIL import Image, ImageDraw
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import prepare_inputs, extract_predictions_to_bboxes
from vlm_fo1.task_templates import (
    OD_template,
    OD_Counting_template,
    REC_template,
    Grounding_template,
)


# Load model once at startup (cached)
print("Loading VLM-FO1 model...")
model_path = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"

tokenizer, model, image_processors = load_pretrained_model(
    model_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    load_8bit=False,  # Set to True for T4 GPU to reduce memory
)

print(f"✓ Model loaded on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"✓ PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")


def detect_objects(
    image: Image.Image,
    query: str,
    task_type: str = "Object Detection",
    confidence_threshold: float = 0.3,
) -> tuple:
    """
    Run VLM-FO1 inference on an image.

    Args:
        image: PIL Image
        query: Object to detect or task query
        task_type: Type of task to perform
        confidence_threshold: Confidence threshold for detections

    Returns:
        (annotated_image, output_text)
    """
    if image is None:
        return None, "Please upload an image"

    # Select task template
    task_templates = {
        "Object Detection": OD_template,
        "Counting": OD_Counting_template,
        "Referring Expression": REC_template,
        "Grounding": Grounding_template,
    }
    template = task_templates.get(task_type, OD_template)

    # Generate dummy bounding boxes (in production, use UPN detector)
    w, h = image.size
    bbox_list = [
        [w * 0.1, h * 0.1, w * 0.4, h * 0.4],
        [w * 0.5, h * 0.5, w * 0.9, h * 0.9],
        [w * 0.2, h * 0.5, w * 0.5, h * 0.8],
    ]

    # Prepare messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "uploaded_image"}},
            {"type": "text", "text": template.format(query)},
        ],
        "bbox_list": bbox_list,
    }]

    # Run inference
    try:
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
        predicted_bboxes = extract_predictions_to_bboxes(outputs, bbox_list)

        # Draw bounding boxes on image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for bbox in predicted_bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), query, fill="red")

        return annotated_image, f"**Model Output:**\n{outputs}\n\n**Detected Boxes:** {len(predicted_bboxes)}"

    except Exception as e:
        return None, f"Error during inference: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="VLM-FO1 Demo") as demo:
    gr.Markdown("""
    # 🔍 VLM-FO1: Fine-Grained Visual Perception

    Upload an image and specify what to detect. The model will find and highlight the objects.

    **Powered by:** VLM-FO1 Qwen2.5-VL-3B
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            query_input = gr.Textbox(
                label="What to detect",
                placeholder="e.g., orange, person, car",
                value="orange"
            )
            task_type = gr.Radio(
                choices=["Object Detection", "Counting", "Referring Expression", "Grounding"],
                value="Object Detection",
                label="Task Type"
            )
            confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="Confidence Threshold"
            )
            submit_btn = gr.Button("Run Detection", variant="primary")

        with gr.Column():
            image_output = gr.Image(type="pil", label="Detection Results")
            text_output = gr.Textbox(label="Model Output", lines=6)

    # Example images
    gr.Examples(
        examples=[
            ["demo/demo_image.jpg", "orange", "Object Detection"],
            ["demo/demo_image.jpg", "person", "Counting"],
        ],
        inputs=[image_input, query_input, task_type],
    )

    # Connect button
    submit_btn.click(
        fn=detect_objects,
        inputs=[image_input, query_input, task_type, confidence],
        outputs=[image_output, text_output]
    )

    gr.Markdown("""
    ---
    **Note:** This demo uses simplified bounding box proposals. For production use,
    integrate with the UPN detector for better results.

    **Links:**
    - [GitHub Repository](https://github.com/om-ai-lab/VLM-FO1)
    - [Paper](https://arxiv.org/abs/2509.25916)
    - [Model Card](https://huggingface.co/omlab/VLM-FO1_Qwen2.5-VL-3B-v01)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
