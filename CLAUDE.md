# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VLM-FO1 is a vision-language model framework that bridges high-level reasoning with fine-grained perception. It operates as a plug-and-play module integrating with pre-trained VLMs (Vision Language Models), specifically built on Qwen2.5-VL-3B. The system combines:

- **Primary Vision Tower**: Qwen2.5-VL encoder for semantic understanding
- **Auxiliary Vision Tower**: DaViT encoder for fine-grained spatial features
- **HFRE Module** (Hybrid Fine-grained Region Encoder): Fuses semantic and perception features
- **UPN Detector**: Optional object proposal network for generating bounding box candidates

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Installation with Extras (using pyproject.toml)
```bash
# Core dependencies only
pip install -e .[base]

# Include UPN detector dependencies
pip install -e .[base,upn]
```

### UPN Native Extension Build (Optional)

The UPN detector uses a CUDA extension (`MultiScaleDeformableAttention`) for optimal performance:

```bash
cd detect_tools/upn/ops
pip install -v -e . --no-build-isolation
```

**Important**: If CUDA headers (`cusparse.h`) are not found, the build is skipped and a CPU fallback is used automatically. The extension requires:
- Pre-installed PyTorch with CUDA support
- CUDA toolkit headers (specifically `cusparse.h`)
- Compiler toolchain (g++/clang)

## Running Inference

### 1. Basic Inference with Pre-defined Bounding Boxes
```bash
python inference.py
```
- Loads model from HuggingFace (`omlab/VLM-FO1_Qwen2.5-VL-3B-v01`) or local path
- Uses manually specified bounding boxes
- Outputs visualization to `demo/vlm_fo1_result.jpg`

### 2. Inference with UPN Object Detector
```bash
# First, download UPN checkpoint
wget https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth -P resources/

# Run inference
python scripts/inference_with_upn.py
```
- Generates object proposals using UPN
- Filters proposals by confidence score (default: 0.3)
- Passes top 100 proposals to VLM-FO1

### 3. Gradio Demo
```bash
python demo/gradio_demo.py
```

## Model Architecture

### Key Components

1. **vlm_fo1/model/omchat_arch.py**: Core meta-model class (`OmChatMetaModel`)
   - Builds vision towers (primary + auxiliary)
   - Instantiates the HFRE module (`HFREModule`)
   - Manages multi-modal projectors

2. **vlm_fo1/model/multimodal_visual_prompt_encoder/hybrid_finegrained_region_encoder.py**:
   - `HFREModule`: Extracts region features using RoI pooling
   - Combines features from both vision towers
   - Applies position embeddings (bbox-based or learnable)

3. **vlm_fo1/model/language_model/omchat_qwen2_5_vl.py**:
   - `OmChatQwen25VLForCausalLM`: Main model class extending Qwen2.5-VL

4. **vlm_fo1/model/builder.py**:
   - `load_pretrained_model()`: Loads model, tokenizer, and image processors
   - Handles vision tower initialization for both primary and auxiliary encoders

### Vision Tower Loading Pattern

The model uses a two-tower architecture:
- **Primary tower** (Qwen2.5-VL): Loaded from model checkpoint, provides semantic features
- **Auxiliary tower** (DaViT): Loaded separately, provides fine-grained spatial features at multiple scales

Both are loaded in `bfloat16` precision by default.

## Task Templates

Located in `vlm_fo1/task_templates.py`. Key templates:
- `OD_template`: Object detection - "Please detect {object} in this image. Answer with object indexes."
- `OD_Counting_template`: Counting objects
- `REC_template`: Referring expression comprehension
- `Grounding_template`: Dense captioning with grounding
- `Region_OCR_template`: OCR for specific regions
- `Viusal_Region_Reasoning_template`: Reasoning with <think> and <answer> tags

## Message Format

The model expects messages with bounding boxes:
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": img_path}},
        {"type": "text", "text": OD_template.format("orange")}
    ],
    "bbox_list": [[x1, y1, x2, y2], ...]  # List of bounding boxes in xyxy format
}]
```

Bounding boxes are in absolute coordinates (not normalized).

## Inference Flow

1. **Prepare inputs**: `vlm_fo1.mm_utils.prepare_inputs()` tokenizes text and processes images
2. **Generate**: `model.generate()` with prepared kwargs
3. **Decode**: Tokenizer decodes output tokens
4. **Extract predictions**: `extract_predictions_to_bboxes()` or `extract_predictions_to_indexes()` parses model output to get object indexes
5. **Visualize**: `draw_bboxes_and_save()` renders bounding boxes on image

## Evaluation

### CountBench/Pixmo-Count
```bash
python evaluation/eval_countbench.py
```

### COCO Detection
```bash
python evaluation/eval_coco.py
```
Outputs predictions in COCO format for use with standard COCO evaluation tools.

## UPN Detector Integration

The UPN (Universal Proposal Network) detector is integrated from ChatRex. Key files:
- `detect_tools/upn/inference_wrapper.py`: `UPNWrapper` class for easy inference
- `detect_tools/upn/configs/upn_large.py`: Model configuration
- `detect_tools/upn/ops/`: CUDA extension for deformable attention (optional, has CPU fallback)

The detector returns proposals with scores; filter using `UPNWrapper.filter(proposals, min_score=threshold)`.

## Device and Precision

- Default device: `cuda`
- Default dtype: `torch.bfloat16`
- Supports quantization: `load_8bit=True` or `load_4bit=True` in `load_pretrained_model()`
- Flash Attention 2 is used for Qwen2.5-VL (`attn_implementation="flash_attention_2"`)

## Common Patterns

### Loading Model
```python
from vlm_fo1.model.builder import load_pretrained_model
tokenizer, model, image_processors = load_pretrained_model(
    "omlab/VLM-FO1_Qwen2.5-VL-3B-v01",
    device="cuda"
)
```

### Running Inference
```python
from vlm_fo1.mm_utils import prepare_inputs
generation_kwargs = prepare_inputs(
    model_path, model, image_processors, tokenizer, messages,
    max_tokens=4096, top_p=0.05, temperature=0.0, do_sample=False
)
with torch.inference_mode():
    output_ids = model.generate(**generation_kwargs)
    outputs = tokenizer.decode(output_ids[0, generation_kwargs['inputs'].shape[1]:])
```

## Model Checkpoints

Pre-trained weights are expected in `resources/` directory:
- VLM-FO1 model: Download from [HuggingFace](https://huggingface.co/omlab/VLM-FO1_Qwen2.5-VL-3B-v01)
- UPN detector: Download `upn_large.pth` from ChatRex GitHub releases

## Package Structure

- `vlm_fo1/`: Core model implementation
  - `model/`: Model architecture (vision towers, projectors, HFRE, language model)
  - `mm_utils.py`: Utilities for tokenization, visualization, input preparation
  - `task_templates.py`: Prompt templates for different tasks
  - `constants.py`: Special tokens and constants
- `detect_tools/upn/`: UPN detector integration
  - `ops/`: CUDA extension with CPU fallback
  - `models/`: UPN architecture (backbone, encoder, decoder)
- `scripts/`: Inference scripts
- `evaluation/`: Benchmark evaluation scripts
- `demo/`: Gradio demo and sample images

## Special Tokens

Defined in `vlm_fo1/constants.py`:
- `DEFAULT_IMAGE_TOKEN`: "<image>"
- `DEFAULT_REGION_TOKEN`: "<region>"
- `DEFAULT_REGION_FEATURE_TOKEN`: "<regionfeat>"
- `IMAGE_TOKEN_INDEX`: -200
- `DEFAULT_REGION_INDEX`: -201
