import modal
import os
import subprocess
from pathlib import Path

# Define the build image
# Must match the inference image's CUDA and Torch versions exactly
build_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "g++", "libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.4.0",
        "packaging",
        "ninja",
        "wheel",
        "setuptools"
    )
)

app = modal.App("flash-attn-builder")

@app.function(
    gpu="A10G",  # GPU required for compilation
    image=build_image,
    timeout=3600  # Give it plenty of time (usually takes ~10-20 mins)
)
def build_flash_attn():
    print("Starting FlashAttention build...")
    
    # Clone specific version
    subprocess.run(
        ["git", "clone", "https://github.com/Dao-AILab/flash-attention.git"], 
        check=True
    )
    
    os.chdir("flash-attention")
    # Checkout a recent stable tag compatible with Torch 2.4
    # v2.6.3 is a good candidate
    subprocess.run(["git", "checkout", "v2.6.3"], check=True)
    
    print("Compiling wheel... this will take a while.")
    # Force CUDA build
    env = os.environ.copy()
    env["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"
    
    try:
        subprocess.run(
            ["python", "setup.py", "bdist_wheel"], 
            check=True,
            env=env,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("Build failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e
    
    # Find the generated wheel
    dist_dir = Path("dist")
    wheel_files = list(dist_dir.glob("*.whl"))
    
    if not wheel_files:
        raise Exception("No wheel file found in dist/")
        
    wheel_path = wheel_files[0]
    print(f"Build complete! Found wheel: {wheel_path.name}")
    
    # Read and return the wheel content and name
    return wheel_path.name, wheel_path.read_bytes()

@app.local_entrypoint()
def main():
    print("Triggering remote build...")
    name, content = build_flash_attn.remote()
    
    output_dir = Path("wheels")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / name
    output_path.write_bytes(content)
    
    print(f"Saved wheel to: {output_path.absolute()}")
