#!/usr/bin/env python3
"""
Enhanced example showing bounding box extraction with coordinates.
"""

import requests
import json

API_URL = "https://animeshraj958--vlm-fo1-inference-vlminference-web-generate.modal.run"

def test_detection(image_url: str, prompt: str):
    """Test the API with detailed output."""
    print(f"\n{'='*70}")
    print(f"🔍 Detecting: '{prompt}'")
    print(f"📷 Image: {image_url}")
    print(f"{'='*70}\n")
    
    payload = {
        "image_url": image_url,
        "prompt": prompt
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        print("✅ API Response:")
        print(json.dumps(result, indent=2))
        
        # Parse the new enhanced response
        if isinstance(result, dict) and "bboxes" in result:
            print(f"\n📊 Summary:")
            print(f"   Detected: {result.get('detected', False)}")
            print(f"   Number of detections: {result.get('num_detections', 0)}")
            print(f"   Image size: {result.get('image_size', {})}")
            
            if result.get('bboxes'):
                print(f"\n📦 Bounding Boxes (x1, y1, x2, y2):")
                for i, bbox in enumerate(result['bboxes']):
                    print(f"   Box {i}: {bbox}")
            else:
                print(f"\n⚠️  No bounding boxes detected")
                print(f"   Raw output: {result.get('raw_output', 'N/A')}")
        else:
            # Old format
            print(f"\n⚠️  Old API format detected")
            print(f"   Result: {result.get('result', result)}")
        
        return result
        
    except requests.exceptions.Timeout:
        print("⏱️  Request timed out (cold start can take 30-60s)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Test 1: Cat image
    test_detection(
        image_url="https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
        prompt="cat"
    )
    
    # Test 2: Person
    test_detection(
        image_url="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
        prompt="person"
    )
    
    # Test 3: Dog
    test_detection(
        image_url="https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800",
        prompt="dog"
    )
