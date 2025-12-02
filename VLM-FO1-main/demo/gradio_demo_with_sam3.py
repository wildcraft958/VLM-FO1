import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_bbox, plot_mask, COLORS
import matplotlib.pyplot as plt

from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    extract_predictions_to_indexes,
)
from vlm_fo1.task_templates import *
import torch
import os
from copy import deepcopy


EXAMPLES = [
    ["demo/sam3_examples/00000-72.jpg","airplane with letter AE on its body"],
    ["demo/sam3_examples/00000-32.jpg","the lying cat which is not black"],
    ["demo/sam3_examples/00000-22.jpg","person wearing a black top"],
    ["demo/sam3_examples/000000378453.jpg", "zebra inside the mud puddle"],
    ["demo/sam3_examples/00000-242.jpg", "person who is holding a book"],
]


def get_valid_examples():
    valid_examples = []
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    for example in EXAMPLES:
        img_path = example[0]
        full_path = os.path.join(demo_dir, img_path)
        if os.path.exists(full_path):
            valid_examples.append([
                full_path,
                example[1], 
                example[2]
            ])
        elif os.path.exists(img_path):
            valid_examples.append([
                img_path,
                example[1], 
                example[2]
            ])
    return valid_examples


def detect_model(image, text, threshold=0.3):
    inference_state = sam3_processor.set_image(image)
    output = sam3_processor.set_text_prompt(state=inference_state, prompt=text)
    boxes, scores, masks = output["boxes"], output["scores"], output["masks"]   
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices][:100, :]
    scores = scores[sorted_indices][:100]
    masks = masks[sorted_indices][:100]
    # If the highest confidence score is greater than 0.5, filter with 0.3 threshold
    if len(scores) > 0 and scores[0] > 0.75:
        conf_threshold = 0.3
        
    else:
        conf_threshold = 0.05
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    masks = masks[mask]
    # Keep boxes with score > 0.8 in a separate list
    high_conf_mask = scores > 0.8
    high_conf_boxes = boxes[high_conf_mask]
    
    print("========boxes========\n", boxes.tolist())
    print("========scores========\n", scores.tolist())
    print("========high_conf_boxes (>0.8)========\n", high_conf_boxes.tolist())

    output = {
        "boxes": boxes,
        "scores": scores,
        "masks": masks,
    }
    return boxes.tolist(), scores.tolist(), high_conf_boxes.tolist(), masks.tolist(), output


def multimodal_model(image, bboxes, scores, text):
    if len(bboxes) == 0:
        return None, {}, []
    
    if '<image>' in text:
        print(text)
        parts = [part.replace('\\n', '\n') for part in re.split(rf'(<image>)', text) if part.strip()]
        print(parts)
        content = []
        for part in parts:
            if part == '<image>':
                content.append({"type": "image_url", "image_url": {"url": image}})
            else:
                content.append({"type": "text", "text": part})
    else:
        content = [{
            "type": "image_url",
            "image_url": {
                "url": image
            }
            }, {
                "type": "text",
                "text": text
            }]

    messages = [
        {
            "role": "user",
            "content": content,
            "bbox_list": bboxes
        }
    ]
    generation_kwargs = prepare_inputs(model_path, model, image_processors, tokenizer, messages,
    max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False, image_size=1024)
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
        outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
        print("========output========\n", outputs)

    if '<ground>' in outputs:
        prediction_dict = extract_predictions_to_indexes(outputs)
    else:
        match_pattern = r"<region(\d+)>"
        matches = re.findall(match_pattern, outputs)
        prediction_dict = {f"<region{m}>": {int(m)} for m in matches}

    ans_bbox_json = []
    ans_bbox_list = []
    for k, v in prediction_dict.items():
        for box_index in v:
            box_index = int(box_index)
            if box_index < len(bboxes):
                current_bbox = bboxes[box_index]
                current_score = scores[box_index]
                ans_bbox_json.append({
                    "region_index": f"<region{box_index}>",
                    "xmin": current_bbox[0],
                    "ymin": current_bbox[1],
                    "xmax": current_bbox[2],
                    "ymax": current_bbox[3],
                    "label": k,
                    "score": current_score
                })
                ans_bbox_list.append(current_bbox)

    return outputs, ans_bbox_json, ans_bbox_list


def draw_bboxes(img, results):
    fig, ax = plt.subplots(figsize=(12, 8))
    # fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img)
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
    ax.axis("off")
    fig.tight_layout(pad=0)
    
    # Convert matplotlib figure to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
    plt.close(fig)
    
    return pil_img



def process(image, prompt, threshold=0):
    if image is None:
        error_msg = "Error: Please upload an image or select a valid example."
        print(f"Error: image is None, original input type: {type(image)}")
        return None, None, error_msg, []
    
    try:
        image = image.convert('RGB')
    except Exception as e:
        error_msg = f"Error: Cannot process image - {str(e)}"
        return None, None, error_msg, []

    bboxes, scores, high_conf_bboxes, masks, output = detect_model(image, prompt, threshold)

    fo1_prompt = OD_Counting_template.format(prompt)
    ans, ans_bbox_json, ans_bbox_list = multimodal_model(image, bboxes, scores, fo1_prompt)

    detection_image = draw_bboxes(image, output)

    annotated_bboxes = []
    if len(ans_bbox_json) > 0:
        img_width, img_height = image.size
        for item in ans_bbox_json:
            xmin = max(0, min(img_width, int(item['xmin'])))
            ymin = max(0, min(img_height, int(item['ymin'])))
            xmax = max(0, min(img_width, int(item['xmax'])))
            ymax = max(0, min(img_height, int(item['ymax'])))
            annotated_bboxes.append(
                ((xmin, ymin, xmax, ymax), item['label'])
            )
    annotated_image = (image, annotated_bboxes)

    return annotated_image, detection_image, ans_bbox_json


def update_btn(is_processing):
    if is_processing:
        return gr.update(value="Processing...", interactive=False)
    else:
        return gr.update(value="Submit", interactive=True)


def launch_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 VLM-FO1 + SAM3 Demo")
        gr.Markdown("""
        ### 📋 Instructions
        Combine the SAM3 detection results with the VLM-FO1 model to enchance its dectection and segmentation performance on complex label tasks.

        **How it works**
        1. Upload or pick an example image.
        2. Describe the target object in natural language.
        3. Hit **Submit** to run SAM3 + VLM-FO1.
        
        **Outputs**
        - `SAM3 Result`: raw detections with masks/bboxes generated by SAM3.
        - `VLM-FO1 Result`: filtered detections plus labels generated by VLM-FO1.
        
        **Tips**
        - One prompt at a time is currently supported. Multiple label prompts will be supported soon.
        - Use the examples below to quickly explore the pipeline.
        """)
        
        gr.Markdown("""
        ### 🔗 References
        - [SAM3](https://github.com/facebookresearch/sam3)
        - [VLM-FO1](https://github.com/om-ai-lab/VLM-FO1)
        """)
        
        with gr.Row():
            with gr.Column():
                img_input_draw = gr.Image(
                    label="Image Input",
                    type="pil",
                    sources=['upload'],
                )

                gr.Markdown("### Prompt")

                prompt_input = gr.Textbox(
                    label="Label Prompt", 
                    lines=2,
                )

                submit_btn = gr.Button("Submit", variant="primary")
                
                    
                examples = gr.Examples(
                    examples=EXAMPLES,
                    inputs=[img_input_draw, prompt_input],
                    label="Click to load example",
                    examples_per_page=5
                )

            with gr.Column():
                with gr.Accordion("SAM3 Result", open=True):
                    image_output_detection = gr.Image(label="SAM3 Result", height=400)

                image_output = gr.AnnotatedImage(label="VLM-FO1 Result", height=400)

                ans_bbox_json = gr.JSON(label="Extracted Detection Output")

        submit_btn.click(
            update_btn, 
            inputs=[gr.State(True)], 
            outputs=[submit_btn], 
            queue=False
        ).then(
            process,
            inputs=[img_input_draw, prompt_input],
            outputs=[image_output, image_output_detection, ans_bbox_json],
            queue=True
        ).then(
            update_btn, 
            inputs=[gr.State(False)], 
            outputs=[submit_btn], 
            queue=False
        )
    
    return demo

if __name__ == "__main__":
    # model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01' 
    # sam3_model_path = './resources/sam3/sam3.pt'

    model_path = 'omlab/VLM-FO1_Qwen2.5-VL-3B-v01'
    tokenizer, model, image_processors = load_pretrained_model(
        model_path=model_path,
        device="cuda:0",
    )
    sam3_model = build_sam3_image_model(device="cuda:0")
    sam3_processor = Sam3Processor(sam3_model, confidence_threshold=0.0, device="cuda:0")

    demo = launch_demo()
    demo.launch()
