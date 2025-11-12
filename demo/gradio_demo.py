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


TASK_TYPES = {
    "OD/REC": OD_template,
    "ODCounting": OD_Counting_template,
    "Region_OCR": "Please provide the ocr results of these regions in the image.",
    "Brief_Region_Caption": "Provide a brief description for these regions in the image.",
    "Detailed_Region_Caption": "Provide a detailed description for these regions in the image.",
    "Grounding": Grounding_template,
    "Viusal_Region_Reasoning": Viusal_Region_Reasoning_template,
}



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

    prediction_dict = extract_predictions_to_indexes(outputs)

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


def extract_bbox_and_original_image(edited_image: dict):
    original_image = edited_image["background"]
    bbox_list = []

    if original_image is None:
        return None, "Error, Please upload an image."

    if edited_image["layers"] is None or len(edited_image["layers"]) == 0:
        return original_image, []

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

    return original_image, bbox_list


def process(image, prompt, threshold):
    image, bbox_list = extract_bbox_and_original_image(image)
    image = image.convert('RGB')

    if len(bbox_list) == 0:
        # Get bboxes from detection model
        bboxes = detect_model(image, threshold)
    else:
        bboxes = bbox_list
        for idx in range(len(bboxes)):
            prompt += f'<region{idx}>'

    ans, ans_bbox_json, ans_bbox_list = multimodal_model(image, bboxes, prompt)


    image_with_opn = draw_bboxes(image, bboxes)

    annotated_bboxes = []
    if len(ans_bbox_json) > 0:
        for item in ans_bbox_json:
            annotated_bboxes.append(
                ((int(item['xmin']), int(item['ymin']), int(item['xmax']), int(item['ymax'])), item['label'])
            )
    annotated_image = (image, annotated_bboxes)

    return annotated_image, image_with_opn, ans, ans_bbox_json


def show_label_input(choice):
    return gr.update(visible=(choice == "OmDet"))


def update_btn(is_processing):
    if is_processing:
        return gr.update(value="Processing...", interactive=False)
    else:
        return gr.update(value="Submit", interactive=True)


def launch_demo():
    with gr.Blocks() as demo:
        gr.Markdown("## VLM-FO1 Demo")
        gr.Markdown("""
        **Instructions:**
        1. Upload an image, then you can either draw circular regions on it using the red brush as the input regions or let the detection model detect the regions for you.
        2. Select a task template and replace the [WRITE YOUR INPUT HERE] with your input targets, or write your own prompt.\n
        For example, if you want to detect "person" and "dog", you can replace the [WRITE YOUR INPUT HERE] with "person, dog".\n
        3. Adjust the detection threshold if needed
        4. Click Submit to get results
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

                task_type_input.change(
                    set_prompt_from_template,
                    inputs=task_type_input,
                    outputs=prompt_input
                )


                threshold_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Detection Model Threshold")
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                with gr.Accordion("Detection Result", open=True):
                    image_output_opn = gr.Image(label="Detection Result")

                image_output = gr.AnnotatedImage(label="Multimodal Model Output", height=500)

                result_output = gr.Textbox(label="Multimodal Model Output")
                ans_bbox_json = gr.JSON(label="Extracted Detection Output")

        submit_btn.click(update_btn, inputs=[gr.State(True)], outputs=[submit_btn], queue=False).then(
            process,
            inputs=[img_input_draw, prompt_input, threshold_input],
            outputs=[image_output, image_output_opn, result_output, ans_bbox_json],
            queue=True
        ).then(update_btn, inputs=[gr.State(False)], outputs=[submit_btn], queue=False)
    
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
    demo.launch(server_name="0.0.0.0", share=False, server_port=8000, debug=True)



