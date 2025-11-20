import { useState, useEffect, useRef } from 'react';

export default function DetectionResults({ results, loading, imageUrl }) {
    const [selectedTab, setSelectedTab] = useState('visual');
    const canvasRef = useRef(null);

    // Draw bounding boxes on canvas
    useEffect(() => {
        if (!results || !imageUrl || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
            // Set canvas size to image size
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw image
            ctx.drawImage(img, 0, 0);

            // Draw bounding boxes
            if (results.detections && results.detections.length > 0) {
                results.detections.forEach((det, idx) => {
                    const [x1, y1, x2, y2] = det.bbox;

                    // Draw box
                    ctx.strokeStyle = '#EF4444'; // red-500
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    // Draw label background
                    ctx.fillStyle = '#EF4444';
                    const label = `${det.label} (${det.region_id})`;
                    const textMetrics = ctx.measureText(label);
                    const textHeight = 20;
                    ctx.fillRect(x1, y1 - textHeight, textMetrics.width + 10, textHeight);

                    // Draw label text
                    ctx.fillStyle = 'white';
                    ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
                    ctx.fillText(label, x1 + 5, y1 - 5);
                });
            }
        };

        img.src = imageUrl.startsWith('data:')
            ? imageUrl
            : imageUrl;
    }, [results, imageUrl]);

    if (loading) {
        return (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <div className="flex flex-col items-center">
                    <svg className="animate-spin h-12 w-12 text-indigo-600 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="text-gray-600">Running detection...</p>
                    <p className="text-sm text-gray-500 mt-2">First request may take 30-60s</p>
                </div>
            </div>
        );
    }

    if (!results) {
        return (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
                <div className="text-gray-400">
                    <svg className="mx-auto h-16 w-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="text-gray-600">No results yet</p>
                    <p className="text-sm text-gray-500 mt-2">Upload an image and click "Run Detection"</p>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
            {/* Tabs */}
            <div className="border-b border-gray-200">
                <div className="flex">
                    <button
                        onClick={() => setSelectedTab('visual')}
                        className={`flex-1 py-3 px-4 text-sm font-medium ${selectedTab === 'visual'
                                ? 'border-b-2 border-indigo-600 text-indigo-600'
                                : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        Visual Results
                    </button>
                    <button
                        onClick={() => setSelectedTab('json')}
                        className={`flex-1 py-3 px-4 text-sm font-medium ${selectedTab === 'json'
                                ? 'border-b-2 border-indigo-600 text-indigo-600'
                                : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        JSON Output
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="p-6">
                {selectedTab === 'visual' && (
                    <div>
                        {/* Summary */}
                        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <p className="text-gray-600">Detected:</p>
                                    <p className="font-semibold text-gray-900">
                                        {results.detected ? '✅ Yes' : '❌ No'}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-gray-600">Detections:</p>
                                    <p className="font-semibold text-gray-900">
                                        {results.num_detections}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-gray-600">Proposals:</p>
                                    <p className="font-semibold text-gray-900">
                                        {results.num_proposals}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-gray-600">Image Size:</p>
                                    <p className="font-semibold text-gray-900">
                                        {results.image_size?.width} × {results.image_size?.height}
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Canvas with bounding boxes */}
                        <div className="mb-4">
                            <canvas
                                ref={canvasRef}
                                className="max-w-full h-auto rounded-lg shadow-md"
                            />
                        </div>

                        {/* Detections List */}
                        {results.detections && results.detections.length > 0 && (
                            <div>
                                <h3 className="text-sm font-medium text-gray-900 mb-2">
                                    Detected Objects:
                                </h3>
                                <div className="space-y-2">
                                    {results.detections.map((det, idx) => (
                                        <div key={idx} className="p-3 bg-gray-50 rounded-md">
                                            <div className="flex justify-between items-start">
                                                <div>
                                                    <p className="font-medium text-gray-900">{det.label}</p>
                                                    <p className="text-xs text-gray-500">Region ID: {det.region_id}</p>
                                                </div>
                                                <span className="text-xs bg-indigo-100 text-indigo-800 px-2 py-1 rounded">
                                                    {idx + 1}
                                                </span>
                                            </div>
                                            <p className="text-xs text-gray-600 mt-1">
                                                BBox: [{det.bbox.map(v => v.toFixed(1)).join(', ')}]
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {selectedTab === 'json' && (
                    <div>
                        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-xs">
                            {JSON.stringify(results, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </div>
    );
}
