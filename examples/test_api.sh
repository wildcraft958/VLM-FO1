#!/bin/bash
# Quick test script for your deployed VLM-FO1 API

API_URL="https://animeshraj958--vlm-fo1-inference-vlminference-web-generate-dev.modal.run"

echo "Testing VLM-FO1 API at: $API_URL"
echo ""

# Test 1: Detect lion
echo "Test 1: Detecting 'lion' in image..."
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://raw.githubusercontent.com/gradio-app/gradio/main/demo/blocks_neural_instrument_coding/files/lion.jpg",
    "prompt": "lion"
  }'
echo -e "\n"

# Test 2: Detect orange
echo "Test 2: Detecting 'orange' in image..."
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://raw.githubusercontent.com/omlab/VLM-FO1/main/image.png",
    "prompt": "orange"
  }'
echo -e "\n"

echo "Tests complete!"
