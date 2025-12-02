import gradio as gr
from PIL import Image
import re
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_bbox, plot_mask, COLORS, show_mask
import matplotlib.pyplot as plt
import cv2
import tempfile
from sam3.model_builder import build_sam3_video_model
from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    extract_predictions_to_indexes,
)
from vlm_fo1.task_templates import *
import torch
import os


# Example videos with prompts and starting frames
EXAMPLES = [
    ["demo/sam3_video_examples/bedroom.mp4", "the feet of little girl", 0],
    ["demo/sam3_video_examples/penguins.mp4", "the closet penguin", 0],
    ["demo/sam3_video_examples/basketball.mp4", "the person shooting a basket", 47],
    ["demo/sam3_video_examples/person.mp4", "a little girl with red cloth", 0],
]


def get_valid_examples():
    """
    Validate example paths and return only those that exist.
    Checks both relative to demo directory and absolute paths.
    """
    valid_examples = []
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    
    for example in EXAMPLES:
        img_path = example[0]
        full_path = os.path.join(demo_dir, img_path)
        
        if os.path.exists(full_path):
            valid_examples.append([full_path, example[1], example[2]])
        elif os.path.exists(img_path):
            valid_examples.append([img_path, example[1], example[2]])
            
    return valid_examples


def detect_model(image, text, threshold=0.3):
    """
    Run SAM3 detection on an image with text prompt.
    Returns bounding boxes, confidence scores, and masks.
    Applies adaptive thresholding based on highest confidence score.
    """
    # Set image and text prompt for SAM3
    inference_state = sam3_processor.set_image(image)
    output = sam3_processor.set_text_prompt(state=inference_state, prompt=text)
    boxes, scores, masks = output["boxes"], output["scores"], output["masks"]
    
    # Sort by confidence and keep top 100
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices][:100, :]
    scores = scores[sorted_indices][:100]
    masks = masks[sorted_indices][:100]
    
    # Adaptive confidence threshold: if highest score > 0.6, use 0.3, otherwise use 0.001
    if len(scores) > 0 and scores[0] > 0.6:
        conf_threshold = 0.3
    else:
        conf_threshold = 0.001
        
    print("========scores========\n", scores)
    
    # Filter by confidence threshold and keep top 50
    mask = scores > conf_threshold
    boxes = boxes[mask][:50, :]
    scores = scores[mask][:50]
    masks = masks[mask][:50]

    # Track high confidence boxes (>0.8) separately
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
    """
    Use VLM-FO1 model to filter and refine detections.
    Takes image, bounding boxes, and text prompt to identify relevant objects.
    Returns model output text, filtered bounding boxes in JSON format, and bbox list.
    """
    if len(bboxes) == 0:
        return None, {}, []
    
    # Parse text prompt and construct message content
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
        content = [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": text}
        ]

    # Prepare messages with bounding boxes
    messages = [
        {
            "role": "user",
            "content": content,
            "bbox_list": bboxes
        }
    ]
    
    # Generate model output
    generation_kwargs = prepare_inputs(
        model_path, model, image_processors, tokenizer, messages,
        max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False, image_size=1024
    )
    
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
        outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:]).strip()
        print("========output========\n", outputs)

    # Extract region predictions from model output
    if '<ground>' in outputs:
        prediction_dict = extract_predictions_to_indexes(outputs)
    else:
        match_pattern = r"<region(\d+)>"
        matches = re.findall(match_pattern, outputs)
        prediction_dict = {f"<region{m}>": {int(m)} for m in matches}

    # Build filtered bounding box results
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
    """
    Visualize detection results by drawing bounding boxes and masks on image.
    Returns a PIL Image with annotations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    
    # Draw each detection with mask and bounding box
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h, w,
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


def get_video_info(video_path, target_frame_idx=None):
    """
    Extract video information and return frame count, target frame image, and slider update.
    If target_frame_idx is provided, attempts to extract that specific frame.
    """
    if video_path is None:
        return gr.update(maximum=0, value=0), None, None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return gr.update(maximum=0, value=0), None, None
    
    # Count total frames and extract target frame
    frame_count = 0
    target_idx = 0
    if target_frame_idx is not None:
        try:
            target_idx = int(target_frame_idx)
        except:
            target_idx = 0

    target_frame_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Capture target frame when index matches
        if frame_count == target_idx:
            target_frame_img = frame
            
        frame_count += 1
    cap.release()
    
    # Fallback to first frame if target frame not found
    if target_frame_img is None and frame_count > 0:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            target_frame_img = frame
            target_idx = 0
    
    if target_frame_img is not None:
        target_frame_img = cv2.cvtColor(target_frame_img, cv2.COLOR_BGR2RGB)
        return gr.update(maximum=max(0, frame_count - 1), value=target_idx), target_frame_img, None
        
    return gr.update(maximum=0, value=0), None, None


def get_frame_preview(video_path, frame_idx):
    """
    Extract and return a specific frame from video for preview.
    """
    if video_path is None:
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def process_detection(video_path, prompt, frame_idx=0, threshold=0):
    """
    Run detection pipeline on selected video frame.
    Combines SAM3 detection with VLM-FO1 filtering.
    Returns visualization images and detection results.
    """
    if video_path is None:
        return None, None, None, 0
    
    # Extract selected frame from video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None, None, None, frame_idx
    cap.release()

    # Convert frame to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run SAM3 detection
    bboxes, scores, high_conf_bboxes, masks, output = detect_model(image_pil, prompt, threshold)
    
    # Run VLM-FO1 filtering
    fo1_prompt = OD_Counting_template.format(prompt)
    ans, ans_bbox_json, ans_bbox_list = multimodal_model(image_pil, bboxes, scores, fo1_prompt)

    # Visualize SAM3 detection results
    detection_image = draw_bboxes(image_pil, output)

    # Prepare VLM-FO1 annotated image
    annotated_bboxes = []
    if len(ans_bbox_json) > 0:
        img_width, img_height = image_pil.size
        for item in ans_bbox_json:
            # Clamp coordinates to image bounds
            xmin = max(0, min(img_width, int(item['xmin'])))
            ymin = max(0, min(img_height, int(item['ymin'])))
            xmax = max(0, min(img_width, int(item['xmax'])))
            ymax = max(0, min(img_height, int(item['ymax'])))
            annotated_bboxes.append(
                ((xmin, ymin, xmax, ymax), item['label'])
            )
    fo1_annotated_image = (image_pil, annotated_bboxes)

    return detection_image, fo1_annotated_image, ans_bbox_json, frame_idx


def process_tracking(video_path, ans_bbox_json, start_frame_idx=0, progress=gr.Progress()):
    """
    Track detected objects throughout the entire video using SAM3 video tracker.
    Returns a video file with tracking visualization.
    """
    if video_path is None or ans_bbox_json is None or len(ans_bbox_json) == 0:
        return None

    # Load video frames for visualization
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame0 = cap.read()
    if not ret:
        return None

    video_frames_for_vis = [cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    total_frames = len(video_frames_for_vis)
    height, width = frame0.shape[:2]

    # Initialize SAM3 video tracker
    inference_state = predictor.init_state(video_path=video_path)
    predictor.clear_all_points_in_video(inference_state)

    # Add detected objects to tracker
    for i, item in enumerate(ans_bbox_json):
        obj_id = i + 1
        box = np.array([[item['xmin'], item['ymin'], item['xmax'], item['ymax']]], dtype=np.float32)
        # Normalize box coordinates
        rel_box = box / np.array([width, height, width, height], dtype=np.float32)
        
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            obj_id=obj_id,
            box=rel_box
        )

    # Propagate tracking through video
    print("Running propagation...")
    video_segments = {}
    
    progress(0, desc="Tracking Forward...")
    
    # Convert cached features to float for compatibility
    cached_features = inference_state['cached_features']
    for frame_idx, (image, backbone_out) in cached_features.items():
        cached_features[frame_idx] = (image.to(torch.float), backbone_out)
    
    # Propagate tracking forward through all frames
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, start_frame_idx=0, max_frame_num_to_track=600, 
        reverse=False, propagate_preflight=True
    ):
        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(obj_ids)
        }
        progress(min(1.0, len(video_segments) / total_frames), 
                desc=f"Tracking Frame {len(video_segments)}/{total_frames}")
            
    print("Propagation completed")

    # Render tracking results to video
    output_video_path = tempfile.mktemp(suffix=".mp4")
    fps = 30
    height, width = video_frames_for_vis[0].shape[:2]
    
    # Use MP4V codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Render each frame with tracking masks
    for i, frame in enumerate(video_frames_for_vis):
        # Update progress periodically
        if i % 5 == 0:
            progress(min(1.0, i / total_frames), desc=f"Rendering Video Frame {i}/{total_frames}")
             
        # Create matplotlib figure for rendering
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.imshow(frame)
        
        # Overlay tracking masks for current frame
        if i in video_segments:
            for out_obj_id, out_mask in video_segments[i].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        
        # Convert matplotlib figure to video frame
        fig = plt.gcf()
        fig.canvas.draw()
        
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_array = img_array[:, :, :3]
        img_array = cv2.resize(img_array, (width, height))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        out_video.write(img_array)
        plt.close("all")

    out_video.release()
    return output_video_path


def update_detect_btn_start():
    """Disable detect button during processing."""
    return gr.update(value="Processing...", interactive=False)


def update_detect_btn_end():
    """Re-enable detect button after processing."""
    return gr.update(value="Detect", interactive=True)


def update_track_btn_start():
    """Disable track button during processing."""
    return gr.update(value="Processing...", interactive=False)


def update_track_btn_end():
    """Re-enable track button after processing."""
    return gr.update(value="Track Video", interactive=True)


def launch_demo():
    """
    Launch Gradio demo interface for VLM-FO1 + SAM3 video tracking.
    """
    with gr.Blocks() as demo:
        # State variables for passing data between steps
        state_bboxes = gr.State()
        state_selected_frame_idx = gr.State(value=0)

        gr.Markdown("# 🚀 VLM-FO1 + SAM3 Video Tracking Demo")
        gr.Markdown("""
        ### 📋 Instructions
        Combine SAM3 detection with VLM-FO1 model to enhance detection and segmentation 
        performance on complex label tasks, then track objects in video.

        **How it works**
        1. Upload or pick an example video.
        2. Describe the target object in natural language.
        3. Select a frame using the slider.
        4. Hit **Detect** to run SAM3 + VLM-FO1 on the selected frame.
        5. Review the results, then hit **Track Video** to track the object throughout the video.
        
        **Tips**
        - One prompt at a time is currently supported.
        """)
        
        gr.Markdown("""
        ### 🔗 References
        - [SAM3](https://github.com/facebookresearch/sam3)
        - [VLM-FO1](https://github.com/om-ai-lab/VLM-FO1)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Video input and frame selection
                img_input_draw = gr.Video(
                    label="Video Input",
                    sources=['upload'],
                    height=200
                )

                frame_slider = gr.Slider(
                    minimum=0, maximum=0, step=1, 
                    label="Select Frame", value=0, interactive=True
                )
                selected_frame_preview = gr.Image(
                    label="Selected Frame Preview", 
                    height=200, interactive=False
                )

                gr.Markdown("### Prompt")

                target_frame_idx_component = gr.Number(
                    label="Selected Frame Index", 
                    value=None, visible=False
                )

                prompt_input = gr.Textbox(
                    label="Label Prompt",
                    lines=2,
                )

                with gr.Row():
                    detect_btn = gr.Button("Detect", variant="primary")
                    track_btn = gr.Button("Track Video", variant="secondary")
                
                # Example videos
                examples = gr.Examples(
                    examples=EXAMPLES,
                    inputs=[img_input_draw, prompt_input, target_frame_idx_component],
                    label="Click to load example",
                    examples_per_page=5
                )

            with gr.Column(scale=2):
                # Detection results display
                with gr.Accordion("Detection Results (Selected Frame)", open=True):
                    with gr.Row():
                        image_output_detection = gr.Image(label="SAM3 Result", height=350)
                        image_output_fo1 = gr.AnnotatedImage(label="VLM-FO1 Result", height=350)

                # Tracking results display
                image_output = gr.Video(label="VLM-FO1 + SAM3 Video Tracking Result", height=400)

        # Event handlers
        
        # Update frame slider and preview when video is uploaded
        img_input_draw.change(
            get_video_info,
            inputs=[img_input_draw, target_frame_idx_component],
            outputs=[frame_slider, selected_frame_preview, target_frame_idx_component]
        )

        # Update frame preview when slider is moved
        frame_slider.release(
            get_frame_preview,
            inputs=[img_input_draw, frame_slider],
            outputs=[selected_frame_preview]
        )

        # Detection button click handler
        detect_btn.click(
            update_detect_btn_start,
            inputs=[],
            outputs=[detect_btn],
            queue=False
        ).then(
            process_detection,
            inputs=[img_input_draw, prompt_input, frame_slider],
            outputs=[image_output_detection, image_output_fo1, state_bboxes, state_selected_frame_idx],
            queue=True
        ).then(
            update_detect_btn_end,
            inputs=[],
            outputs=[detect_btn],
            queue=False
        )

        # Tracking button click handler
        track_btn.click(
            update_track_btn_start,
            inputs=[],
            outputs=[track_btn],
            queue=False
        ).then(
            process_tracking,
            inputs=[img_input_draw, state_bboxes, state_selected_frame_idx],
            outputs=[image_output],
            queue=True
        ).then(
            update_track_btn_end,
            inputs=[],
            outputs=[track_btn],
            queue=False
        )
    
    return demo


if __name__ == "__main__":
    # Model paths
    model_path = './resources/VLM-FO1_Qwen2.5-VL-3B-v01'
    sam3_model_path = 'resources/sam3/sam3.pt'

    # Load VLM-FO1 model
    tokenizer, model, image_processors = load_pretrained_model(
        model_path=model_path,
        device="cuda:0",
    )
    
    # Load SAM3 image model
    sam3_model = build_sam3_image_model(checkpoint_path=sam3_model_path, device="cuda")
    sam3_processor = Sam3Processor(sam3_model, confidence_threshold=0.0, device="cuda")

    # Load SAM3 video model
    sam3_video_model = build_sam3_video_model(checkpoint_path=sam3_model_path, device="cuda")
    predictor = sam3_video_model.tracker
    predictor.backbone = sam3_video_model.detector.backbone

    # Launch demo
    demo = launch_demo()
    # demo.launch(server_name="0.0.0.0", share=False, server_port=8000, debug=False)
    demo.launch()
