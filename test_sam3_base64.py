#!/usr/bin/env python3
"""
Test script for SAM3 endpoint with base64 image input.
This script helps debug issues with base64 image submission.

Usage:
    python test_sam3_base64.py <image_path> <query> [api_url]
    
Example:
    python test_sam3_base64.py demo.jpg "the ball nearest to the bear"
"""

import sys
import base64
import requests
import json
import time

# Default API URL - replace with your Modal deployment URL
# Format: https://YOUR-WORKSPACE--vlm-fo1-inference-web-api.modal.run
DEFAULT_API_URL = "https://YOUR-WORKSPACE--vlm-fo1-inference-web-api.modal.run"

def image_to_base64(image_path):
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_echo(api_url):
    """Test if the API is reachable."""
    print("\n" + "="*60)
    print("TEST 1: Echo endpoint (testing API connectivity)")
    print("="*60)
    try:
        response = requests.post(
            f"{api_url}/echo",
            json={"test": "data", "number": 123},
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_base64_validation(api_url, image_path):
    """Test base64 image validation."""
    print("\n" + "="*60)
    print("TEST 2: Base64 validation endpoint")
    print("="*60)
    try:
        image_base64 = image_to_base64(image_path)
        print(f"Image encoded: {len(image_base64)} chars")
        
        response = requests.post(
            f"{api_url}/test_base64",
            json={"image_base64": image_base64},
            timeout=30
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False

def test_sam3_inference(api_url, image_path, query):
    """Test SAM3 inference with base64 image."""
    print("\n" + "="*60)
    print("TEST 3: SAM3 inference with base64 image")
    print("="*60)
    try:
        image_base64 = image_to_base64(image_path)
        print(f"Image encoded: {len(image_base64)} chars")
        print(f"Query: {query}")
        
        payload = {
            "image_base64": image_base64,
            "query": query,
            "confidence_threshold": 0.5,
            "max_proposals": 100
        }
        
        print(f"\nSending request to: {api_url}/web_inference_sam3")
        print("This may take 30-60s on first request (cold start)...")
        start_time = time.time()
        
        response = requests.post(
            f"{api_url}/web_inference_sam3",
            json=payload,
            timeout=180  # 3 minutes timeout
        )
        
        elapsed = time.time() - start_time
        print(f"\nRequest completed in {elapsed:.1f}s")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ SUCCESS!")
            print(f"Raw Output: {result.get('raw_output', 'N/A')}")
            print(f"Detected: {result.get('detected', False)}")
            print(f"Number of detections: {result.get('num_detections', 0)}")
            
            if result.get('detections'):
                print("\nDetections:")
                for i, det in enumerate(result['detections'], 1):
                    print(f"  {i}. Label: {det['label']}")
                    print(f"     BBox: {det['bbox']}")
            
            if result.get('sam3_proposals'):
                print(f"\nSAM3 Proposals: {result['sam3_proposals'].get('count', 0)}")
            
            print(f"\nImage Size: {result.get('image_size', 'N/A')}")
            return True
        else:
            print(f"\n❌ ERROR: Status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out (model may be cold-starting or stuck)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response: {json.dumps(e.response.json(), indent=2)}")
            except:
                print(f"Response: {e.response.text}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_sam3_base64.py <image_path> <query> [api_url]")
        print("\nExample:")
        print('  python test_sam3_base64.py demo.jpg "the ball nearest to the bear"')
        print("\nOr with custom API URL:")
        print('  python test_sam3_base64.py demo.jpg "the ball nearest to the bear" https://your-workspace--vlm-fo1-inference-web-api.modal.run')
        sys.exit(1)
    
    image_path = sys.argv[1]
    query = sys.argv[2]
    api_url = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_API_URL
    
    if api_url == DEFAULT_API_URL:
        print("⚠ WARNING: Using default API URL. Please provide your Modal deployment URL as the third argument.")
        print(f"Current URL: {api_url}")
        print("\nTo find your URL:")
        print("  1. Deploy with: modal deploy examples/modal_app.py")
        print("  2. Modal will show the URL, or check: modal app list")
        response = input("\nContinue with default URL? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print(f"\nAPI URL: {api_url}")
    print(f"Image: {image_path}")
    print(f"Query: {query}")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_echo(api_url):
        tests_passed += 1
    else:
        print("\n⚠ Echo test failed. API may not be reachable. Check your URL.")
        return
    
    if test_base64_validation(api_url, image_path):
        tests_passed += 1
    else:
        print("\n⚠ Base64 validation failed. Check your image file.")
        return
    
    if test_sam3_inference(api_url, image_path, query):
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"Tests completed: {tests_passed}/{total_tests} passed")
    print("="*60)

if __name__ == "__main__":
    main()

