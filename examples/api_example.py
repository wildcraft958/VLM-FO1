#!/usr/bin/env python3
"""
Example script demonstrating how to call the deployed VLM-FO1 Modal API.

Usage:
    python api_example.py
"""

import requests
import json

# Your actual deployed Modal URL
MODAL_API_URL = "https://animeshraj958--vlm-fo1-inference-vlminference-web-generate-dev.modal.run"

def call_vlm_inference(image_url: str, prompt: str):
    """
    Call the VLM-FO1 inference API.
    
    Args:
        image_url: URL to an image (publicly accessible)
        prompt: Object detection prompt (e.g., "person", "car", "dog")
    
    Returns:
        dict: API response containing the grounding result
    """
    payload = {
        "image_url": image_url,
        "prompt": prompt
    }
    
    headers = {
        "Content-Type": "application/json",
        # Uncomment below if your Modal app requires authentication:
        # "Authorization": f"Bearer {YOUR_MODAL_TOKEN}"
    }
    
    try:
        response = requests.post(MODAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None

if __name__ == "__main__":
    # Example 1: Detect "orange"
    print("Example 1: Detecting 'orange'")
    result = call_vlm_inference(
        image_url="https://raw.githubusercontent.com/omlab/VLM-FO1/main/image.png",
        prompt="orange"
    )
    if result:
        print(f"Result: {result['result']}\n")
    
    # Example 2: Detect "person"
    print("Example 2: Detecting 'person'")
    result = call_vlm_inference(
        image_url="https://example.com/street-photo.jpg",
        prompt="person"
    )
    if result:
        print(f"Result: {result['result']}\n")
    
    # Example 3: Multiple objects
    print("Example 3: Detecting 'car and tree'")
    result = call_vlm_inference(
        image_url="https://example.com/outdoor-scene.jpg",
        prompt="car and tree"
    )
    if result:
        print(f"Result: {result['result']}\n")
