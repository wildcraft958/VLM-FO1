import { useState } from 'react';
import ImageInput from './components/ImageInput';
import SettingsPanel from './components/SettingsPanel';
import DetectionResults from './components/DetectionResults';
import { detectObjects } from './utils/api';

function App() {
  const [imageUrl, setImageUrl] = useState('');
  const [prompt, setPrompt] = useState('cat');
  const [threshold, setThreshold] = useState(0.3);
  const [taskType, setTaskType] = useState('Object Detection');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleDetection = async () => {
    if (!imageUrl) {
      setError('Please provide an image');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await detectObjects(imageUrl, prompt, threshold);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to detect objects');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            🔍 VLM-FO1 Object Detection
          </h1>
          <p className="mt-2 text-sm text-gray-600">
            Powered by UPN + VLM-FO1 on Modal A100 GPUs
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Input */}
          <div className="space-y-6">
            <ImageInput imageUrl={imageUrl} setImageUrl={setImageUrl} />
            <SettingsPanel
              prompt={prompt}
              setPrompt={setPrompt}
              threshold={threshold}
              setThreshold={setThreshold}
              taskType={taskType}
              setTaskType={setTaskType}
              onDetect={handleDetection}
              loading={loading}
            />

            {/* Disclaimer */}
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-yellow-800">
                    Privacy Notice
                  </h3>
                  <div className="mt-2 text-sm text-yellow-700">
                    <p>
                      • Uploaded images are converted to base64 and sent directly to the Modal API
                    </p>
                    <p>
                      • Images are processed in memory and not stored
                    </p>
                    <p>
                      • First request may take 30-60s (cold start)
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div>
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="flex">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <p className="ml-3 text-sm text-red-700">{error}</p>
                </div>
              </div>
            )}

            <DetectionResults
              results={results}
              loading={loading}
              imageUrl={imageUrl}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
