import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

# Cell 1: Installation
code_install = """
!pip install "numpy<2.3"
!pip install vllm xformers
!pip install opencv-python matplotlib pandas pyarrow tqdm aiohttp
!pip install git+https://github.com/facebookresearch/sam2.git
# Download SAM2 checkpoint
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O sam2.1_hiera_large.pt
"""

# Cell 2: Imports and Setup
code_imports = """
import subprocess
import os
import signal
import time
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import asyncio
import base64
import aiohttp
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from typing import List, Optional

# Constants
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
SAM2_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
API_KEY = "EMPTY"
API_BASE = "http://localhost:8000/v1"
"""

# Cell 3: vLLM Server Management
code_vllm_server = """
# Start vLLM server in background
def start_vllm_server(model_name):
    command = [
        "nohup",
        "env", "VLLM_ATTENTION_BACKEND=XFORMERS",
        "vllm", "serve",
        model_name,
        "--trust-remote-code",
        "--port", "8000",
        "--gpu-memory-utilization", "0.6",
        "--max-model-len", "32768",
        "--block-size", "64"
    ]
    
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    
    vllm_log = open('vllm.log', 'w')
    process = subprocess.Popen(
        command,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setpgrp
    )
    print(f"vLLM server started with PID: {process.pid}")
    return process

def wait_for_server(timeout=1200):
    start = time.time()
    while True:
        diff = time.time() - start
        if diff > timeout:
            print("Timeout waiting for vLLM server.")
            return False
        
        try:
            # Check if server is responding
            import requests
            response = requests.get(f"{API_BASE.replace('/v1', '')}/health")
            if response.status_code == 200:
                print("vLLM server is ready!")
                return True
        except Exception:
            time.sleep(10)
            print(f"Waiting for server... {int(diff)}s")

def kill_server(process):
    if process:
        pid = process.pid
        try:
            os.killpg(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to process group {pid}.")
            time.sleep(2)
            try:
                os.killpg(pid, signal.SIGKILL)
                print(f"Sent SIGKILL to process group {pid}.")
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            print("Process already gone.")

# Start the server
vllm_process = start_vllm_server(MODEL_NAME)
is_ready = wait_for_server()
if not is_ready:
    print("Server failed to start. Checking logs:")
    with open("vllm.log", "r") as f:
        print(f.read())
    kill_server(vllm_process)
    raise RuntimeError("vLLM server failed to start")
"""

# Cell 4: Data Loading
code_data = """
# Load dataset from parquet
import pandas as pd
import json

# Load the parquet file
df = pd.read_parquet("train-00000-of-00001.parquet")
print(f"Loaded {len(df)} rows from parquet")
print(f"Columns: {df.columns.tolist()}")

# The parquet file has 'objects' as a JSON string that needs parsing
# Each row represents an image with potentially multiple objects
# We'll parse the 'objects' field to get individual samples
data = []
for idx, row in df.iterrows():
    # Parse the objects JSON string
    if 'objects' in row and row['objects']:
        try:
            objects_list = json.loads(row['objects'])
            
            # Each object becomes a separate sample
            for obj in objects_list:
                sample = {
                    'image_id': row.get('image_id', f"img_{idx}"),
                    'image_path': row.get('image_path', ''),
                    'target_object': {
                        'referring_sentence': obj.get('referring_sentence', ''),
                        'obj_coord': obj.get('obj_coord', [0, 0, 1, 1]),
                        'obj_cls': obj.get('obj_cls', ''),
                        'obj_id': obj.get('obj_id', 0)
                    }
                }
                data.append(sample)
        except Exception as e:
            print(f"Error parsing objects for row {idx}: {e}")
            continue

print(f"Processed into {len(data)} individual object samples")
"""

# Cell 5: Async Inference Function
code_inference = """
# 1. Initialize the Asynchronous OpenAI client
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# 2. Async helper to encode image
async def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 3. Worker function for async inference
async def _process_one_image(
    image_path: str,
    text_prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    pbar: tqdm,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    system_prompt = "You are a helpful assistant. Solve the user's query" if system_prompt is None else system_prompt
    
    async with semaphore:
        try:
            base64_image = await encode_image_to_base64(image_path)
            
            # Create a fresh client instance for each request to ensure async works
            async_client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
            
            response = await async_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.01,
            )
            
            output = response.choices[0].message.content
            return output

        except Exception as e:
            print(f"❌ Error for sample{index + 1}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            pbar.update(1)

# 4. Main async function
async def infer_async(
    image_paths: List[str],
    text_prompts: List[str],
    concurrency: int,
    system_prompts: Optional[List[str]] = None
) -> List[Optional[str]]:
    if len(text_prompts) != len(image_paths):
        raise ValueError("Length mismatch")
    
    if system_prompts is None:
        system_prompts = [None] * len(text_prompts)

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(image_paths), desc="Processing images")
    
    tasks = [
        _process_one_image(img_path, text_prompt, semaphore, i, pbar, system_prompt)
        for i, (img_path, text_prompt, system_prompt) in enumerate(zip(image_paths, text_prompts, system_prompts))
    ]
    
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results

def denormalize_bbox(bbox_norm, width, height):
    '''Convert normalized [x1, y1, x2, y2] to pixel coordinates'''
    x1, y1, x2, y2 = bbox_norm
    return [
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height)
    ]
"""

# Cell 6: SAM2 Initialization
code_sam2_init = """
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 initialized")
"""

# Cell 7: OpenCV and IoU Utils
code_utils = """
def return_box(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get axis-aligned bounding box directly
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, x + w, y + h]
    return None

def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union
"""

# Cell 8: Main Pipeline Loop
code_pipeline = """
# Prepare data for batch processing
image_paths = []
text_prompts = []
system_prompts = []
ground_truths = []  # will store pixel coords
sample_indices = []

# System prompt for object detection via referring expressions
detection_system_prompt = (
    "You are a helpful assistant for object detection. "
    "Given an image and a referring expression describing a specific object, "
    "return ONLY the bounding box for that object as a simple list in the format: "
    "[x1, y1, x2, y2] where coordinates are normalized floats between 0.0 and 1.0. "
    "Do not include any explanation, just the four numbers in brackets."
)

# Process first 5 samples
NUM_SAMPLES = 5
for idx, sample in enumerate(data[:NUM_SAMPLES]):
    # Get image path and fix relative paths
    img_path = sample['image_path']
    if img_path.startswith('./'):
        img_path = '/root/' + img_path[2:]
    elif not img_path.startswith('/'):
        img_path = '/root/' + img_path
    
    # Get referring sentence from target_object
    referring_sentence = sample['target_object']['referring_sentence']
    
    # Get ground truth bbox (normalized coords)
    gt_bbox_norm = sample['target_object']['obj_coord']  # [x1, y1, x2, y2]
    
    image_paths.append(img_path)
    text_prompts.append(f"Locate: {referring_sentence}\\nReturn bounding box as [x1, y1, x2, y2]")
    system_prompts.append(detection_system_prompt)
    ground_truths.append(gt_bbox_norm)  # Store normalized for now
    sample_indices.append(idx)

print(f"Starting async inference for {len(image_paths)} samples...")

# Run Async Inference
results_json = await infer_async(image_paths, text_prompts, concurrency=5, system_prompts=system_prompts)

results = []

# Process results
for i, (img_path, content, gt_bbox_norm) in enumerate(zip(image_paths, results_json, ground_truths)):
    print(f"\\n{'='*60}")
    print(f"Processing sample {i}...")
    print(f"{'='*60}")
    
    if content is None:
        print(f"⚠️  Skipping sample {i} due to inference failure.")
        continue
    
    # Show raw vLLM output for debugging
    print(f"📝 VLM Response: {content[:200]}...")
    
    # Load image to get dimensions
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to load image: {img_path}")
        continue
    height, width, _ = image.shape
    print(f"📐 Image dimensions: {width}x{height}")
    
    # Parse vLLM output for bbox
    vllm_bbox_norm = None
    try:
        # Extract bbox from response (looking for [x1, y1, x2, y2] pattern)
        import re
        
        # Try to find normalized float bbox pattern
        bbox_pattern = r'\\[\\s*([0-9]*\\.?[0-9]+)\\s*,\\s*([0-9]*\\.?[0-9]+)\\s*,\\s*([0-9]*\\.?[0-9]+)\\s*,\\s*([0-9]*\\.?[0-9]+)\\s*\\]'
        match = re.search(bbox_pattern, content)
        
        if match:
            vllm_bbox_norm = [float(match.group(i)) for i in range(1, 5)]
            print(f"✅ Parsed bbox (normalized): {vllm_bbox_norm}")
            
            # Check if values are actually normalized (0-1 range)
            if any(v > 1.5 or v < -0.5 for v in vllm_bbox_norm):
                print(f"⚠️  Bbox values seem out of range, attempting to normalize...")
                # If values are large, assume they might be pixel coords
                if any(v > 10 for v in vllm_bbox_norm):
                    print(f"   Assuming pixel coordinates, converting to normalized...")
                    vllm_bbox_norm = [
                        vllm_bbox_norm[0] / width,
                        vllm_bbox_norm[1] / height,
                        vllm_bbox_norm[2] / width,
                        vllm_bbox_norm[3] / height
                    ]
                    print(f"   Normalized to: {vllm_bbox_norm}")
        else:
            print(f"❌ Could not parse bbox with regex")
            # Try JSON parsing as fallback
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
            else:
                json_content = content.strip()
            
            # Try to parse as JSON array
            parsed = json.loads(json_content)
            if isinstance(parsed, list) and len(parsed) == 4:
                vllm_bbox_norm = [float(v) for v in parsed]
                print(f"✅ Parsed bbox from JSON: {vllm_bbox_norm}")
            else:
                raise ValueError("Could not parse bbox from JSON")
                
    except Exception as e:
        print(f"❌ Error parsing VLM output for sample {i}: {e}")
        print(f"   Full content: {content}")
        continue
    
    # Denormalize both bboxes to pixel coordinates
    vllm_bbox = denormalize_bbox(vllm_bbox_norm, width, height)
    gt_bbox = denormalize_bbox(gt_bbox_norm, width, height)
    
    print(f"📍 vLLM bbox (pixels): {vllm_bbox}")
    print(f"📍 GT bbox (pixels): {gt_bbox}")
    
    # SAM2 Segmentation using vLLM bbox as prompt
    predictor.set_image(image)
    
    # Convert [x1, y1, x2, y2] to SAM2 format
    input_box = np.array([vllm_bbox])
    
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False
    )
    
    # Refine bbox using SAM2 mask
    sample_results = {
        "sample_id": sample_indices[i],
        "image_id": data[sample_indices[i]]['image_id'],
        "referring_sentence": data[sample_indices[i]]['target_object']['referring_sentence'],
        "vllm_bbox": vllm_bbox,
        "sam2_refined_bbox": None,
        "ground_truth_bbox": gt_bbox,
        "iou_vllm_vs_gt": 0.0,
        "iou_sam2_vs_gt": 0.0
    }
    
    if len(masks) > 0:
        # Use the first (and likely only) mask
        refined_box = return_box(masks[0].squeeze())
        
        if refined_box is not None:
            sample_results["sam2_refined_bbox"] = refined_box
            
            # Calculate IoUs
            iou_vllm = calculate_iou(vllm_bbox, gt_bbox)
            iou_sam2 = calculate_iou(refined_box, gt_bbox)
            
            sample_results["iou_vllm_vs_gt"] = iou_vllm
            sample_results["iou_sam2_vs_gt"] = iou_sam2
            
            print(f"📊 vLLM IoU: {iou_vllm:.4f}")
            print(f"📊 SAM2 IoU: {iou_sam2:.4f}")
        else:
            print(f"⚠️  Could not extract refined bbox from mask")
    else:
        print(f"⚠️  No masks generated")
            
    results.append(sample_results)

# Save results
with open("pipeline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\\n{'='*60}")
print(f"✅ Pipeline completed!")
print(f"{'='*60}")
print(f"📁 Results saved to pipeline_results.json")
print(f"📊 Processed {len(results)}/{NUM_SAMPLES} samples successfully")
"""

# Cell 9: Cleanup
code_cleanup = """
kill_server(vllm_process)
print("vLLM server stopped.")
"""

# Add cells to notebook
nb.cells.append(nbf.v4.new_code_cell(code_install))
nb.cells.append(nbf.v4.new_code_cell(code_imports))
nb.cells.append(nbf.v4.new_code_cell(code_vllm_server))
nb.cells.append(nbf.v4.new_code_cell(code_data))
nb.cells.append(nbf.v4.new_code_cell(code_inference))
nb.cells.append(nbf.v4.new_code_cell(code_sam2_init))
nb.cells.append(nbf.v4.new_code_cell(code_utils))
nb.cells.append(nbf.v4.new_code_cell(code_pipeline))
nb.cells.append(nbf.v4.new_code_cell(code_cleanup))

# Write notebook
with open('vllm_sam2_pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook vllm_sam2_pipeline.ipynb created successfully.")
