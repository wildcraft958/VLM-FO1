import axios from 'axios';

// Update these URLs after deploying modal_app.py
// Get the URLs from: modal app list vlm-fo1-inference
const API_URL_BASE = 'https://YOUR-WORKSPACE--vlm-fo1-inference-web-inference.modal.run';
const API_URL_SAM3 = 'https://YOUR-WORKSPACE--vlm-fo1-inference-web-inference-sam3.modal.run';

// Legacy endpoint (from modal_inference.py)
const API_URL_LEGACY = 'https://animeshraj958--vlm-fo1-inference-vlminference-web-generate.modal.run';

export const detectObjects = async (imageUrl, prompt, threshold = 0.3, useSam3 = false, confidenceThreshold = 0.5) => {
    try {
        const endpoint = useSam3 ? API_URL_SAM3 : API_URL_BASE;
        
        const payload = useSam3 ? {
            image_url: imageUrl,
            query: prompt,
            confidence_threshold: confidenceThreshold,
            max_proposals: 100
        } : {
            image_url: imageUrl,
            query: prompt
        };

        const response = await axios.post(endpoint, payload, {
            timeout: 180000 // 3 minutes (SAM3 takes longer)
        });

        return response.data;
    } catch (error) {
        if (error.code === 'ECONNABORTED') {
            throw new Error('Request timed out. The model might be cold-starting (takes 30-60s on first request).');
        }
        if (error.response) {
            throw new Error(`API Error: ${error.response.data?.detail || error.response.statusText}`);
        }
        throw error;
    }
};

// Legacy function for backward compatibility
export const detectObjectsLegacy = async (imageUrl, prompt, threshold = 0.3) => {
    try {
        const response = await axios.post(API_URL_LEGACY, {
            image_url: imageUrl,
            prompt: prompt,
            threshold: threshold
        }, {
            timeout: 120000 // 2 minutes
        });

        return response.data;
    } catch (error) {
        if (error.code === 'ECONNABORTED') {
            throw new Error('Request timed out. The model might be cold-starting (takes 30-60s on first request).');
        }
        throw error;
    }
};

export const fileToDataUrl = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsDataURL(file);
    });
};
