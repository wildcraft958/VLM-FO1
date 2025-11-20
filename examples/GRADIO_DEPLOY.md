# VLM-FO1 Gradio on Modal

## Quick Start

Deploy the Gradio web interface:
```bash
modal deploy examples/modal_gradio.py
```

This will give you a **public URL** for the interactive demo!

## What You Get

- **Interactive Web UI**: Upload images and get instant detections
- **Task Selection**: Choose between Object Detection, Counting, or Grounding
- **Visual Results**: See both proposals and final detections
- **JSON Output**: Get structured bounding box data

## Features

- ✅ Image upload
- ✅ Real-time object detection
- ✅ Bounding box visualization  
- ✅ Adjustable detection threshold
- ✅ Multiple task types
- ✅ Example prompts

## Deploy vs Serve

**Deploy** (persistent, public URL):
```bash
modal deploy examples/modal_gradio.py
```

**Serve** (temporary, for testing):
```bash
modal serve examples/modal_gradio.py
```

## Expected URL

After deployment:
```
https://yourworkspace--vlm-fo1-gradio.modal.run
```

Visit this URL in your browser to use the demo!
