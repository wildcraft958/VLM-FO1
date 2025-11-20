#!/usr/bin/env python3
"""
Test Modal API with the user's uploaded vehicle image.
"""

import requests
import json

API_URL = "https://animeshraj958--vlm-fo1-inference-vlminference-web-generate.modal.run"

# IMPORTANT: Use the DIRECT image URL, not the page URL!
# Wrong: https://ibb.co/fGnwMzBy (this is the webpage)
# Right: https://i.ibb.co/[ID]/[filename].png (this is the actual image)
# 
# To get it: Visit your ImgBB link → Look for "Direct link" → Copy that URL
# 
# For now, using a test image:
IMAGE_URL = "https://i.postimg.cc/XqqBqX65/test2.jpg"
def test_vehicle_detection():
    print("=" * 70)
    print("🚗 Testing VLM-FO1 API - Vehicle Detection")
    print("=" * 70)
    print(f"\n📷 Image: {IMAGE_URL}")
    print(f"💬 Prompt: 'vehicle'")
    print(f"⚙️  Threshold: 0.1")
    print("-" * 70)
    
    payload = {
        "image_url": IMAGE_URL,
        "prompt": "Please find a windmill located at the left bottom of the image",
        "threshold": 0.14
    }
    
    try:
        print("\n⏳ Calling API... (this may take 30-60s on first request)")
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.ok:
            result = response.json()
            
            print(f"\n✅ SUCCESS!\n")
            print(json.dumps(result, indent=2))
            
            print(f"\n📊 SUMMARY:")
            print(f"   Detected: {result.get('detected', False)}")
            print(f"   Number of detections: {result.get('num_detections', 0)}")
            print(f"   Number of proposals: {result.get('num_proposals', 0)}")
            print(f"   Image size: {result.get('image_size', {})}")
            
            if result.get('detections'):
                print(f"\n📦 DETECTIONS:")
                for i, det in enumerate(result['detections'], 1):
                    bbox = det['bbox']
                    print(f"\n   Detection {i}:")
                    print(f"      Label: {det['label']}")
                    print(f"      Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    print(f"      Region ID: {det['region_id']}")
                    print(f"      Confidence: {det.get('confidence', 1.0)}")
                    
                    # Calculate position
                    img_width = result['image_size']['width']
                    img_height = result['image_size']['height']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Determine position
                    h_pos = "left" if center_x < img_width/3 else "right" if center_x > 2*img_width/3 else "center"
                    v_pos = "top" if center_y < img_height/3 else "bottom" if center_y > 2*img_height/3 else "middle"
                    
                    print(f"      Position: {v_pos}-{h_pos}")
            else:
                print(f"\n⚠️  No vehicles detected")
                print(f"   Raw output: {result.get('raw_output', 'N/A')}")
                
            print(f"\n🔍 Raw model output: {result.get('raw_output', 'N/A')}")
            
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n⏱️  Request timed out!")
        print("   (Cold start can take up to 60 seconds)")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_vehicle_detection()
