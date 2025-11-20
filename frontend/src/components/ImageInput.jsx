import { useState } from 'react';
import { fileToDataUrl } from '../utils/api';

export default function ImageInput({ imageUrl, setImageUrl }) {
    const [inputMode, setInputMode] = useState('url'); // 'url' or 'upload'
    const [preview, setPreview] = useState(null);
    const [uploading, setUploading] = useState(false);

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        setUploading(true);
        try {
            const dataUrl = await fileToDataUrl(file);
            setImageUrl(dataUrl);
            setPreview(dataUrl);
        } catch (error) {
            alert('Failed to load image');
        } finally {
            setUploading(false);
        }
    };

    const handleURLChange = (e) => {
        const url = e.target.value;
        setImageUrl(url);
        if (url) {
            setPreview(url);
        }
    };

    return (
        <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
                📷 Image Input
            </h2>

            {/* Mode Selector */}
            <div className="flex space-x-2 mb-4">
                <button
                    onClick={() => setInputMode('url')}
                    className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${inputMode === 'url'
                            ? 'bg-indigo-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                >
                    Image URL
                </button>
                <button
                    onClick={() => setInputMode('upload')}
                    className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${inputMode === 'upload'
                            ? 'bg-indigo-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                >
                    Upload File
                </button>
            </div>

            {/* URL Input */}
            {inputMode === 'url' && (
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Image URL
                    </label>
                    <input
                        type="url"
                        value={imageUrl.startsWith('data:') ? '' : imageUrl}
                        onChange={handleURLChange}
                        placeholder="https://example.com/image.jpg"
                        className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                    <p className="mt-2 text-xs text-gray-500">
                        Paste a direct link to an image file
                    </p>
                </div>
            )}

            {/* File Upload */}
            {inputMode === 'upload' && (
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Upload Image
                    </label>
                    <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-indigo-400 transition-colors">
                        <div className="space-y-1 text-center">
                            <svg
                                className="mx-auto h-12 w-12 text-gray-400"
                                stroke="currentColor"
                                fill="none"
                                viewBox="0 0 48 48"
                            >
                                <path
                                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                                    strokeWidth={2}
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                            </svg>
                            <div className="flex text-sm text-gray-600">
                                <label className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none">
                                    <span>{uploading ? 'Loading...' : 'Upload a file'}</span>
                                    <input
                                        type="file"
                                        className="sr-only"
                                        accept="image/*"
                                        onChange={handleFileUpload}
                                        disabled={uploading}
                                    />
                                </label>
                                <p className="pl-1">or drag and drop</p>
                            </div>
                            <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Preview */}
            {preview && (
                <div className="mt-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Preview:</p>
                    <img
                        src={preview}
                        alt="Preview"
                        className="max-h-64 mx-auto rounded-lg shadow-md"
                        onError={() => setPreview(null)}
                    />
                </div>
            )}
        </div>
    );
}
