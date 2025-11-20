# VLM-FO1 Modal Deployment

This directory contains Modal deployment scripts for VLM-FO1.

## Files

- **`modal_inference.py`** - Production API deployment with UPN integration
- **`modal_gradio.py`** - Gradio web UI (experimental, has ASGI compatibility issues)
- **`test_bboxes.py`** - API testing script with example usage

## Quick Start

### Deploy the API

```bash
modal deploy examples/modal_inference.py
```

This deploys a REST API endpoint for VLM-FO1 object detection.

### Test the API

```bash
python examples/test_bboxes.py
```

### API Usage

```python
import requests

response = requests.post(
    "https://YOUR-WORKSPACE--vlm-fo1-inference-vlminference-web-generate.modal.run",
    json={
        "image_url": "https://example.com/image.jpg",
        "prompt": "cat",
        "threshold": 0.3
    }
)

result = response.json()
# Returns: {"detected": true, "detections": [{"bbox": [x1, y1, x2, y2], "label": "cat", ...}], ...}
```

## How It Works

1. **UPN (Universal Proposal Network)** generates ~100 region proposals
2. **VLM-FO1** identifies which regions match your text prompt  
3. API returns the selected regions with bounding box coordinates

## Features

- ✅ Real bounding box coordinates
- ✅ Multi-object detection
- ✅ Adjustable detection threshold
- ✅ Modal Volume caching (models persist across runs)
- ✅ A100 GPU acceleration

## Notes

- First request: 30-60s (cold start + UPN download)
- Subsequent requests: ~2-5s
- UPN checkpoint (~200MB) downloads on first run
