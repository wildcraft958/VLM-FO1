export default function SettingsPanel({
    prompt,
    setPrompt,
    threshold,
    setThreshold,
    taskType,
    setTaskType,
    onDetect,
    loading
}) {
    const taskTypes = ['Object Detection', 'Object Counting', 'Grounding'];

    return (
        <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
                ⚙️ Settings
            </h2>

            {/* Prompt */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Prompt
                </label>
                <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g., cat, person, vehicle"
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
                <p className="mt-1 text-xs text-gray-500">
                    What objects to detect
                </p>
            </div>

            {/* Task Type */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Task Type
                </label>
                <select
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                >
                    {taskTypes.map((type) => (
                        <option key={type} value={type}>
                            {type}
                        </option>
                    ))}
                </select>
            </div>

            {/* Threshold */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Detection Threshold: {threshold.toFixed(2)}
                </label>
                <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>More detections</span>
                    <span>Higher confidence</span>
                </div>
            </div>

            {/* Detect Button */}
            <button
                onClick={onDetect}
                disabled={loading}
                className={`w-full py-3 px-4 rounded-md font-medium text-white transition-colors ${loading
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800'
                    }`}
            >
                {loading ? (
                    <span className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Detecting...
                    </span>
                ) : (
                    '🚀 Run Detection'
                )}
            </button>
        </div>
    );
}
