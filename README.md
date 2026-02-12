# 🎨 Kolam AI Generator

A beautiful web interface for generating intricate Kolam patterns using AI. This application combines the power of Stable Diffusion with a modern, responsive React frontend to create stunning traditional South Indian art.

## ✨ Features

- **Beautiful Modern UI**: Clean, responsive design with glassmorphism effects
- **AI-Powered Generation**: Uses Stable Diffusion to create detailed Kolam patterns
- **Smart Prompting**: Base prompt always includes "a highly detailed SKS kolam, intricate symmetrical pattern, masterpiece, sharp focus"
- **Quick Presets**: Pre-defined prompt suggestions for common Kolam styles
- **Advanced Settings**: Customizable parameters (steps, CFG scale, sampler, etc.)
- **Image History**: View and download your recent generations
- **Download Support**: Save your generated Kolams as PNG files
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

1. **Stable Diffusion WebUI** with API enabled
2. **Python 3.8+**
3. **Node.js 16+** and npm

### Installation

1. **Clone or download this project**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install React dependencies:**
   ```bash
   cd sd-frontend
   npm install
   ```

4. **Start Stable Diffusion WebUI with API:**
   ```bash
   python launch.py --api --listen
   ```

### Running the Application

#### Option 1: Using the startup script (Recommended)
```bash
python start_server.py
```

#### Option 2: Manual setup

1. **Start the backend server:**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the React frontend (in a new terminal):**
   ```bash
   cd sd-frontend
   npm start
   ```

3. **Open your browser:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 🎯 How to Use

1. **Enter your prompt**: Add additional details to the base Kolam prompt (optional)
2. **Use presets**: Click on quick preset buttons for common styles
3. **Adjust settings**: Expand "Advanced Settings" to customize generation parameters
4. **Generate**: Click "Generate Kolam" to create your artwork
5. **Download**: Hover over generated images and click the download button
6. **View history**: Scroll down to see your recent generations

## 🛠️ Configuration

### Backend Configuration

The FastAPI backend (`app.py`) connects to your local Stable Diffusion WebUI. Key settings:

- **Base Prompt**: Always included: "a highly detailed SKS kolam, intricate symmetrical pattern, masterpiece, sharp focus"
- **Default Parameters**: 50 steps, 512x512 resolution, CFG scale 7.0, Euler sampler
- **Image Storage**: Generated images are saved in the `output/` directory

### Frontend Configuration

The React frontend (`sd-frontend/src/App.js`) provides the user interface:

- **API Endpoint**: Connects to `http://localhost:8000/generate`
- **Responsive Design**: Adapts to different screen sizes
- **Image History**: Keeps last 10 generated images in memory

## 📁 Project Structure

```
kolam-image-gen/
├── app.py                 # FastAPI backend server
├── start_server.py        # Startup script
├── requirements.txt       # Python dependencies
├── output/               # Generated images storage
├── sd-frontend/          # React frontend
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   ├── App.css       # Styling
│   │   └── index.css     # Global styles
│   ├── package.json      # Node.js dependencies
│   └── public/           # Static assets
└── README.md             # This file
```

## 🎨 Customization

### Adding New Preset Prompts

Edit the `presetPrompts` array in `sd-frontend/src/App.js`:

```javascript
const presetPrompts = [
  "traditional South Indian design",
  "geometric mandala pattern",
  "floral border design",
  "peacock feather inspired",
  "lotus flower pattern",
  "temple architecture inspired",
  "your custom prompt here"
];
```

### Modifying the Base Prompt

Update the `base_prompt` variable in `app.py`:

```python
base_prompt = "your custom base prompt here"
```

### Changing Default Settings

Modify the default settings in `sd-frontend/src/App.js`:

```javascript
const [settings, setSettings] = useState({
  steps: 50,           // Number of sampling steps
  width: 512,          // Image width
  height: 512,         // Image height
  cfg_scale: 7.0,      // CFG scale
  sampler_index: "Euler" // Sampler method
});
```

## 🔧 Troubleshooting

### Common Issues

1. **"Stable Diffusion WebUI is not running"**
   - Make sure you started WebUI with `--api` flag
   - Check that WebUI is running on `http://127.0.0.1:7860`

2. **"Failed to generate image"**
   - Check your Stable Diffusion model is loaded
   - Verify WebUI API is accessible
   - Check console for detailed error messages

3. **Frontend not connecting to backend**
   - Ensure backend is running on port 8000
   - Check CORS settings in `app.py`
   - Verify no firewall blocking the connection

4. **Images not saving**
   - Check `output/` directory exists and is writable
   - Verify sufficient disk space

### Performance Tips

- **Reduce steps** for faster generation (20-30 steps often sufficient)
- **Lower resolution** (256x256) for quicker results
- **Use efficient samplers** like Euler or DPM++ 2M
- **Close other applications** to free up GPU memory

## 📝 API Documentation

The FastAPI backend provides several endpoints:

- `POST /generate` - Generate a new Kolam image
- `GET /images/{filename}` - Retrieve a saved image
- `GET /health` - Health check endpoint

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Stable Diffusion** for the AI image generation capabilities
- **FastAPI** for the robust backend framework
- **React** for the modern frontend framework
- **Traditional Kolam artists** for the beautiful art form that inspired this project

---

**Happy Kolam Creating! 🎨✨**
