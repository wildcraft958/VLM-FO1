# VLM-FO1 Deployment Examples

This directory contains platform-specific examples for deploying VLM-FO1 with **unified dependencies** that work reliably across all platforms.

## 🎯 Unified Dependency Management

All examples use the **same dependency versions** defined in `pyproject.toml`, ensuring:
- ✅ **No version conflicts** between platforms
- ✅ **Reproducible builds** everywhere
- ✅ **Easy debugging** - same environment everywhere
- ✅ **Zero compatibility issues**

## 📦 Available Examples

### 1. Modal AI (`modal_app.py`)

Deploy VLM-FO1 on Modal with GPU support.

**Quick Start:**
```bash
# Install Modal
pip install modal

# Deploy
modal deploy examples/modal_app.py

# Run inference
modal run examples/modal_app.py::run_inference --image-path "path/to/image.jpg"
```

**Features:**
- Automatic GPU allocation (T4/A10G/A100)
- Model caching with volumes
- Scalable inference
- Pay-per-use pricing

---

### 2. Jupyter Notebook (`jupyter_notebook.ipynb`)

Interactive notebook for experimentation and development.

**Works On:**
- Local Jupyter Lab/Notebook
- Google Colab
- Kaggle Notebooks
- SageMaker Studio
- Any Jupyter environment

**Quick Start:**
```bash
# Local
jupyter notebook examples/jupyter_notebook.ipynb

# Google Colab
# Upload the notebook to Colab and run!
```

**Features:**
- Automatic CUDA detection
- Step-by-step inference guide
- GPU memory profiling
- Visualization tools

---

### 3. Hugging Face Spaces (`huggingface_space_app.py`)

Deploy a public demo on Hugging Face Spaces.

**Quick Start:**
1. Create a new Space on https://huggingface.co/spaces
2. Upload these files:
   - `app.py` ← `huggingface_space_app.py`
   - `requirements.txt` ← `requirements_hf_spaces.txt`
3. Set hardware to **GPU T4** or better
4. Deploy!

**Example Space:** https://huggingface.co/spaces/P3ngLiu/VLM-FO1-3B-Demo

**Features:**
- Public web interface
- Free GPU hosting (T4)
- Gradio UI
- Instant deployment

---

## 🔑 Key Dependency Specifications

All examples use these **exact versions** (from `pyproject.toml`):

```python
# PyTorch (flexible within bounds)
torch>=2.1,<2.7
torchvision>=0.16

# Core libraries
transformers>=4.45,<5.0
timm>=0.9.0
accelerate>=1.0
safetensors>=0.4.0
pillow>=9.0
numpy>=1.21
```

**Why flexible ranges?**
- Allows minor updates for bug fixes
- Prevents breaking changes from major versions
- Tested compatible range

---

## 🚀 Platform-Specific Notes

### Modal AI
- **GPU Options:** T4 (budget), A10G (balanced), A100 (performance)
- **Cold Start:** ~30s first run, <5s after warming
- **Scaling:** Automatic based on load
- **Cost:** ~$0.60/hour for T4

### Jupyter Notebooks
- **Local:** Requires NVIDIA GPU with CUDA 11.8+
- **Colab:** Free T4 GPU available (limited hours)
- **Kaggle:** 30h/week free GPU time
- **Memory:** Use `load_8bit=True` for constrained GPUs

### Hugging Face Spaces
- **Free Tier:** Community GPU (limited availability)
- **Pro Tier:** Dedicated GPU ($1/month for T4)
- **Persistent:** Automatically restarts on crashes
- **Custom Domain:** Available on Pro tier

---

## 🧪 Testing Your Deployment

### 1. Basic Import Test
```python
import vlm_fo1
print(f"Version: {vlm_fo1.__version__}")
```

### 2. GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### 3. Full Diagnostic
```python
from vlm_fo1._backend import info
import json
print(json.dumps(info(), indent=2))
```

### 4. Quick Inference Test
```python
from vlm_fo1.model.builder import load_pretrained_model
tokenizer, model, image_processors = load_pretrained_model(
    "omlab/VLM-FO1_Qwen2.5-VL-3B-v01",
    device="cuda"
)
print("✓ Model loaded successfully!")
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
**Solution:**
```python
# Use 8-bit quantization
tokenizer, model, image_processors = load_pretrained_model(
    model_path,
    load_8bit=True,  # Reduces memory by ~50%
    device="cuda"
)
```

### "Wheel ABI mismatch"
**Solution:**
```bash
# Install correct PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Reinstall VLM-FO1
BUILD_CUDA=cu118 pip install --force-reinstall git+https://github.com/om-ai-lab/VLM-FO1.git
```

### "Module not found: vlm_fo1"
**Solution:**
```bash
# Ensure installation completed
pip list | grep vlm-fo1

# If not found, reinstall
pip install git+https://github.com/om-ai-lab/VLM-FO1.git
```

### Platform-Specific Issues

**Modal:**
- Increase timeout if model loading fails: `@app.function(timeout=900)`
- Check volume mounting: `/models` should be writable

**Jupyter:**
- Restart kernel if imports fail
- Clear GPU cache: `torch.cuda.empty_cache()`

**Hugging Face Spaces:**
- Check logs in Settings → Logs
- Ensure GPU is enabled (not CPU)
- Model files must be accessible (public repo or add HF token)

---

## 📚 Additional Resources

- **Main README:** [../README.md](../README.md)
- **GPU Development Guide:** [../CONTRIBUTING_GPU.md](../CONTRIBUTING_GPU.md)
- **Diagnostic Tool:** `python tools/diagnose_cuda.py`
- **Selfcheck:** `python -m vlm_fo1`

---

## 💡 Best Practices

1. **Always use unified dependencies** from `pyproject.toml`
2. **Run diagnostics first** before reporting issues
3. **Use 8-bit quantization** on limited GPUs (T4, Colab)
4. **Cache models** when possible (Modal volumes, Colab Drive)
5. **Test locally first** before deploying to cloud platforms

---

## 🤝 Contributing

Found a deployment issue or have a new example?

1. Run diagnostics: `python -m vlm_fo1 > diagnostics.txt`
2. Open an issue: https://github.com/om-ai-lab/VLM-FO1/issues
3. Include your platform, Python version, and diagnostics.txt

---

**Happy deploying! 🚀**
