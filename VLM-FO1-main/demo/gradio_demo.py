import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
from detect_tools.upn import UPNWrapper
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    extract_predictions_to_indexes,
)
from vlm_fo1.task_templates import *
import torch
import os
from copy import deepcopy


TASK_TYPES = {
    "OD/REC": OD_template,
    "ODCounting": OD_Counting_template,
    "Region_OCR": "Please provide the ocr results of these regions in the image.",
    "Brief_Region_Caption": "Provide a brief description for these regions in the image.",
    "Detailed_Region_Caption": "Provide a detailed description for these regions in the image.",
    "Viusal_Region_Reasoning": Viusal_Region_Reasoning_template,
    "OD_All": OD_All_template,
    "Grounding": Grounding_template,
}

EXAMPLES = [
    ["demo_image.jpg", TASK_TYPES["OD/REC"].format("orange, apple"), "OD/REC"],
    ["demo_image_01.jpg", TASK_TYPES["ODCounting"].format("airplane with only one propeller"), "ODCounting"],
    ["demo_image_02.jpg", TASK_TYPES["OD/REC"].format("the ball closest to the bear"), "OD/REC"],
    ["demo_image_03.jpg", TASK_TYPES["OD_All"].format(""), "OD_All"],
    ["demo_image_03.jpg", TASK_TYPES["Viusal_Region_Reasoning"].format("What's the brand of this computer?"), "Viusal_Region_Reasoning"],
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


def detect_model(image, threshold=0.3):
    proposals = upn_model.inference(image)
    filtered_proposals = upn_model.filter(proposals, min_score=threshold)
    return filtered_proposals['original_xyxy_boxes'][0][:100]


def multimodal_model(image, bboxes, text):
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
    max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False)
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
                ans_bbox_json.append({
                    "region_index": f"<region{box_index}>",
                    "xmin": current_bbox[0],
                    "ymin": current_bbox[1],
                    "xmax": current_bbox[2],
                    "ymax": current_bbox[3],
                    "label": k
                })
                ans_bbox_list.append(current_bbox)

    return outputs, ans_bbox_json, ans_bbox_list


def draw_bboxes(image, bboxes, labels=None):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=3)
    return image


def extract_bbox_and_original_image(edited_image):
    """Extract original image and bounding boxes from ImageEditor output"""
    if edited_image is None:
        return None, []
    
    if isinstance(edited_image, dict):
        original_image = edited_image.get("background")
        bbox_list = []

        if original_image is None:
            return None, []

        if edited_image.get("layers") is None or len(edited_image.get("layers", [])) == 0:
            return original_image, []

        try:
            drawing_layer = edited_image["layers"][0]
            alpha_channel = drawing_layer.getchannel('A')
            alpha_np = np.array(alpha_channel)

            binary_mask = alpha_np > 0

            structuring_element = disk(5)
            dilated_mask = binary_dilation(binary_mask, structuring_element)

            labeled_image = label(dilated_mask)
            regions = regionprops(labeled_image)

            for prop in regions:
                y_min, x_min, y_max, x_max = prop.bbox
                bbox_list.append((x_min, y_min, x_max, y_max))
        except Exception as e:
            print(f"Error extracting bboxes from layers: {e}")
            return original_image, []

        return original_image, bbox_list
    elif isinstance(edited_image, Image.Image):
        return edited_image, []
    else:
        print(f"Unknown input type: {type(edited_image)}")
        return None, []


def process(image, example_image, prompt, threshold):
    image, bbox_list = extract_bbox_and_original_image(image)

    if example_image is not None:
        image = example_image
    
    if image is None:
        error_msg = "Error: Please upload an image or select a valid example."
        print(f"Error: image is None, original input type: {type(image)}")
        return None, None, error_msg, []
    
    try:
        image = image.convert('RGB')
    except Exception as e:
        error_msg = f"Error: Cannot process image - {str(e)}"
        return None, None, error_msg, []

    if len(bbox_list) == 0:
        bboxes = detect_model(image, threshold)
    else:
        bboxes = bbox_list
        for idx in range(len(bboxes)):
            prompt += f'<region{idx}>'

    ans, ans_bbox_json, ans_bbox_list = multimodal_model(image, bboxes, prompt)

    image_with_detection = draw_bboxes(image, bboxes)

    annotated_bboxes = []
    if len(ans_bbox_json) > 0:
        for item in ans_bbox_json:
            annotated_bboxes.append(
                ((int(item['xmin']), int(item['ymin']), int(item['xmax']), int(item['ymax'])), item['label'])
            )
    annotated_image = (image, annotated_bboxes)

    return annotated_image, image_with_detection, ans, ans_bbox_json


def update_btn(is_processing):
    if is_processing:
        return gr.update(value="Processing...", interactive=False)
    else:
        return gr.update(value="Submit", interactive=True)


def launch_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 VLM-FO1 Demo")
        gr.Markdown("""
        ### 📋 Instructions
        
        **Step 1: Prepare Your Image**
        - Upload an image using the image editor below
        - *Optional:* Draw circular regions with the red brush to specify areas of interest
        - *Alternative:* If not drawing regions, the detection model will automatically identify regions
        
        **Step 2: Configure Your Task**
        - Select a task template from the dropdown menu
        - Replace `[WRITE YOUR INPUT HERE]` with your target objects or query
        - *Example:* For detecting "person" and "dog", replace with: `person, dog`
        - *Or:* Write your own custom prompt
        
        **Step 3: Fine-tune Detection** *(Optional)*
        - Adjust the detection threshold slider to control sensitivity
        
        **Step 4: Generate Results**
        - Click the **Submit** button to process your request
        - View the detection results and model outputs below
        
        🔗 [GitHub Repository](https://github.com/om-ai-lab/VLM-FO1)
        """)
        
        with gr.Row():
            with gr.Column():
                img_input_draw = gr.ImageEditor(
                    label="Image Input",
                    image_mode="RGBA",
                    type="pil",
                    sources=['upload'],
                    brush=gr.Brush(colors=["#FF0000"], color_mode="fixed", default_size=2),
                    interactive=True
                )

                gr.Markdown("### Prompt & Parameters")

                def set_prompt_from_template(selected_task):
                    return gr.update(value=TASK_TYPES[selected_task].format("[WRITE YOUR INPUT HERE]"))
                
                def load_example(prompt_input, task_type_input, hidden_image_box):
                    cached_image = deepcopy(hidden_image_box)
                    w, h = cached_image.size
    
                    transparent_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))

                    new_editor_value = {
                        "background": cached_image,
                        "layers": [transparent_layer],
                        "composite": None
                    }
                    
                    return new_editor_value, prompt_input, task_type_input
                
                def reset_hidden_image_box():
                    return gr.update(value=None)

                task_type_input = gr.Dropdown(
                    choices=list(TASK_TYPES.keys()),
                    value="OD/REC",
                    label="Prompt Templates",
                    info="Select the prompt template for the task, or write your own prompt."
                )

                prompt_input = gr.Textbox(
                    label="Task Prompt", 
                    value=TASK_TYPES["OD/REC"].format("[WRITE YOUR INPUT HERE]"),
                    lines=2,
                )

                task_type_input.select(
                    set_prompt_from_template,
                    inputs=task_type_input,
                    outputs=prompt_input
                )

                hidden_image_box = gr.Image(label="Image", type="pil", image_mode="RGBA", visible=False)

                threshold_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Detection Model Threshold")
                submit_btn = gr.Button("Submit", variant="primary")
                
                valid_examples = get_valid_examples()
                if len(valid_examples) > 0:
                    gr.Markdown("### Examples")
                    gr.Markdown("Click on the examples below to quickly load images and corresponding prompts:")
                    
                    examples_data = [[example[0], example[1], example[2]] for index, example in enumerate(valid_examples)]
                    
                    examples = gr.Examples(
                        examples=examples_data,
                        inputs=[hidden_image_box, prompt_input, task_type_input],
                        label="Click to load example",
                        examples_per_page=5
                    )
                    
                    examples.load_input_event.then(
                        fn=load_example,
                        inputs=[prompt_input, task_type_input, hidden_image_box], 
                        outputs=[img_input_draw, prompt_input, task_type_input]
                    )

                    img_input_draw.upload(
                        fn=reset_hidden_image_box,
                        outputs=[hidden_image_box]
                    )

            with gr.Column():
                with gr.Accordion("Detection Result", open=True):
                    image_with_detection = gr.Image(label="Detection Result", height=200)

                image_output = gr.AnnotatedImage(label="VLM-FO1 Result", height=400)

                result_output = gr.Textbox(label="VLM-FO1 Output", lines=5)
                ans_bbox_json = gr.JSON(label="Extracted Detection Output")

        submit_btn.click(
            update_btn, 
            inputs=[gr.State(True)], 
            outputs=[submit_btn], 
            queue=False
        ).then(
            process,
            inputs=[img_input_draw, hidden_image_box, prompt_input, threshold_input],
            outputs=[image_output, image_with_detection, result_output, ans_bbox_json],
            queue=True
        ).then(
            update_btn, 
            inputs=[gr.State(False)], 
            outputs=[submit_btn], 
            queue=False
        )
    
    return demo

if __name__ == "__main__":
    model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01' 
    upn_ckpt_path = "./resources/upn_large.pth" 
    tokenizer, model, image_processors = load_pretrained_model(
        model_path=model_path,
        device="cuda:0",
    )
    upn_model = UPNWrapper(upn_ckpt_path)

    demo = launch_demo()
    demo.launch(server_name="0.0.0.0", share=False, server_port=8000, debug=False)
