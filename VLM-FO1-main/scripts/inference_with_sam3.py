import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    extract_predictions_to_indexes,
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
)
from vlm_fo1.task_templates import OD_template

# Paths to required files
sam3_model_path = "./resources/sam3/sam3.pt"  # SAM3 model checkpoint
img_path = "demo/demo_image_02.jpg"             # Path to input image
model_path = 'omlab/VLM-FO1_Qwen2.5-VL-3B-v01'
# model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'  # VLM FO1 model path

label_prompt = "the ball nearest to the bear"

confidence_threshold = 0.5

# Initialize UPN object detector
sam3_model = build_sam3_image_model(checkpoint_path=sam3_model_path, device="cuda")
sam3_processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold, device="cuda")

# Load and preprocess image
img_pil = Image.open(img_path).convert("RGB")

# Run SAM3 to get fine-grained object proposals
inference_state = sam3_processor.set_image(img_pil)
sam3_processor.reset_all_prompts(inference_state)
output = sam3_processor.set_text_prompt(state=inference_state, 
prompt=label_prompt)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
# Sort by scores from high to low
sorted_indices = torch.argsort(scores, descending=True)
masks = masks[sorted_indices][:100, :]
boxes = boxes[sorted_indices][:100, :]
scores = scores[sorted_indices][:100]

# Prepare chat messages with vision input and bounding boxes
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
                "text": OD_template.format(label_prompt),
            },
        ],
        "bbox_list": boxes.tolist(),
    }
]

# Load vision-language model and tokenizer
tokenizer, model, image_processors = load_pretrained_model(model_path)

# Prepare input for model generation
generation_kwargs = prepare_inputs(
    model_path, model, image_processors, tokenizer, messages,
    max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
)

# Run inference and decode output
with torch.inference_mode():
    output_ids = model.generate(**generation_kwargs)
    outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
    print("========output========\n", outputs)

# Convert output prediction (indexes) to bounding box coordinates
bbox_indexes = extract_predictions_to_indexes(outputs)

res = {
}
res_masks = []
for label, index in bbox_indexes.items():
    if label not in res:
        res[label] = []
    for i in index:
        res[label].append(boxes[i].tolist())
        res_masks.append(masks[i].tolist())

# Draw detected bounding boxes and save visualization
draw_bboxes_and_save(
    image=img_pil,
    fo1_bboxes=res,
    output_path="demo/vlm_fo1_result_with_sam3.jpg"
)