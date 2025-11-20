import axios from 'axios';

const API_URL = 'https://animeshraj958--vlm-fo1-inference-vlminference-web-generate.modal.run';

export const detectObjects = async (imageUrl, prompt, threshold = 0.3) => {
    try {
        const response = await axios.post(API_URL, {
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
