# VLM-FO1 React Frontend

Modern React + Vite frontend for VLM-FO1 object detection, deployable to Vercel.

## Features

- ✅ Image upload (converts to base64)
- ✅ URL input support
- ✅ Real-time bbox visualization
- ✅ Adjustable detection threshold
- ✅ Task type selection
- ✅ JSON output view
- ✅ Mobile responsive
- ✅ Privacy-focused (no image storage)

## Local Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

Visit http://localhost:3000

## Deploy to Vercel

### Option 1: Vercel CLI

```bash
npm i -g vercel
vercel
```

### Option 2: Vercel Dashboard

1. Push code to GitHub
2. Go to https://vercel.com/new
3. Import your repository
4. Vercel auto-detects Vite config
5. Deploy!

## How It Works

### Image Upload Flow

1. **User uploads image** → Converted to base64 data URL
2. **Base64 sent to Modal API** → Embedded in request JSON
3. **Modal API decodes base64** → Processes with VLM-FO1 + UPN
4. **Returns bbox coordinates** → Frontend draws boxes on canvas

### Privacy

- Images converted to base64 client-side
- Sent directly to Modal API (no intermediate storage)
- Processed in memory only
- Never stored on servers

## Architecture

```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageInput.jsx       # URL input + file upload
│   │   ├── SettingsPanel.jsx    # Prompt, threshold, task
│   │   └── DetectionResults.jsx # Bbox visualization + JSON
│   ├── utils/
│   │   └── api.js               # Modal API client
│   ├── App.jsx                   # Main app
│   └── main.jsx                  # Entry point
├── package.json
├── vite.config.js
└── vercel.json
```

## Environment

No environment variables needed! The API URL is hardcoded.

To change the API endpoint, edit `src/utils/api.js`:
```javascript
const API_URL = 'YOUR_MODAL_API_URL';
```

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## Notes

- First API request takes 30-60s (cold start)
- Supports PNG, JPG, GIF images
- Max file size: 10MB (adjust as needed)
- Canvas renders bboxes directly on image
