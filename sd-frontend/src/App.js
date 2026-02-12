import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [userPrompt, setUserPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("blurry, deformed, text, watermark, ugly");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [settings, setSettings] = useState({
    steps: 50,
    width: 512,
    height: 512,
    cfg_scale: 7.0,
    sampler_index: "Euler"
  });

  const fileInputRef = useRef(null);

  const generateImage = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userPrompt,
          negative_prompt: negativePrompt,
          ...settings
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate image");
      }

      const data = await response.json();
      const imageData = `data:image/png;base64,${data.image}`;
      
      setImage(imageData);
      setGeneratedImages(prev => [{
        id: Date.now(),
        image: imageData,
        prompt: data.prompt,
        timestamp: data.timestamp,
        filename: data.filename
      }, ...prev.slice(0, 9)]); // Keep last 10 images
      
    } catch (err) {
      console.error("Error generating image:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = (imageData, filename) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename || 'kolam_generated.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const clearHistory = () => {
    setGeneratedImages([]);
    setImage(null);
  };

  const presetPrompts = [
    "traditional South Indian design",
    "geometric mandala pattern",
    "floral border design",
    "peacock feather inspired",
    "lotus flower pattern",
    "temple architecture inspired"
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            <span className="title-icon">🎨</span>
            Kolam AI Generator
          </h1>
          <p className="subtitle">Create beautiful, intricate Kolam patterns with AI</p>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          <div className="generator-section">
            <div className="input-panel">
              <div className="input-group">
                <label className="input-label">
                  <span className="label-icon">✨</span>
                  Additional Prompt
                </label>
                <textarea
                  className="prompt-input"
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  placeholder="Describe your desired kolam pattern (optional)"
                  rows={3}
                />
                <div className="base-prompt">
                  <strong>Base prompt:</strong> "a highly detailed SKS kolam, intricate symmetrical pattern, masterpiece, sharp focus"
                </div>
              </div>

              <div className="preset-prompts">
                <label className="input-label">Quick Presets:</label>
                <div className="preset-buttons">
                  {presetPrompts.map((preset, index) => (
                    <button
                      key={index}
                      className="preset-btn"
                      onClick={() => setUserPrompt(preset)}
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </div>

              <div className="advanced-section">
                <button
                  className="advanced-toggle"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  <span className="toggle-icon">{showAdvanced ? '▼' : '▶'}</span>
                  Advanced Settings
                </button>
                
                {showAdvanced && (
                  <div className="advanced-settings">
                    <div className="setting-row">
                      <label>Negative Prompt:</label>
                      <input
                        type="text"
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                        className="setting-input"
                      />
                    </div>
                    <div className="setting-row">
                      <label>Steps:</label>
                      <input
                        type="number"
                        value={settings.steps}
                        onChange={(e) => setSettings(prev => ({...prev, steps: parseInt(e.target.value)}))}
                        className="setting-input"
                        min="1"
                        max="150"
                      />
                    </div>
                    <div className="setting-row">
                      <label>CFG Scale:</label>
                      <input
                        type="number"
                        value={settings.cfg_scale}
                        onChange={(e) => setSettings(prev => ({...prev, cfg_scale: parseFloat(e.target.value)}))}
                        className="setting-input"
                        min="1"
                        max="20"
                        step="0.1"
                      />
                    </div>
                    <div className="setting-row">
                      <label>Sampler:</label>
                      <select
                        value={settings.sampler_index}
                        onChange={(e) => setSettings(prev => ({...prev, sampler_index: e.target.value}))}
                        className="setting-input"
                      >
                        <option value="Euler">Euler</option>
                        <option value="DPM++ 2M">DPM++ 2M</option>
                        <option value="DPM++ SDE">DPM++ SDE</option>
                        <option value="Heun">Heun</option>
                        <option value="DDIM">DDIM</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              <button
                className={`generate-btn ${loading ? 'loading' : ''}`}
                onClick={generateImage}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Generating...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🎨</span>
                    Generate Kolam
                  </>
                )}
              </button>

              {error && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {error}
                </div>
              )}
            </div>

            <div className="output-panel">
              <div className="image-container">
                {image ? (
                  <div className="generated-image-wrapper">
                    <img src={image} alt="Generated Kolam" className="generated-image" />
                    <div className="image-overlay">
                      <button
                        className="download-btn"
                        onClick={() => downloadImage(image, `kolam_${Date.now()}.png`)}
                      >
                        <span className="btn-icon">💾</span>
                        Download
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="placeholder">
                    <div className="placeholder-icon">🎨</div>
                    <p>Your generated kolam will appear here</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {generatedImages.length > 0 && (
            <div className="history-section">
              <div className="history-header">
                <h3>Recent Generations</h3>
                <button className="clear-btn" onClick={clearHistory}>
                  Clear History
                </button>
              </div>
              <div className="history-grid">
                {generatedImages.map((item) => (
                  <div key={item.id} className="history-item">
                    <img src={item.image} alt="Generated Kolam" className="history-image" />
                    <div className="history-overlay">
                      <button
                        className="history-download-btn"
                        onClick={() => downloadImage(item.image, item.filename)}
                      >
                        💾
                      </button>
                    </div>
                    <div className="history-info">
                      <p className="history-prompt">{item.prompt.split(', ')[1] || 'Base kolam'}</p>
                      <p className="history-time">{new Date(item.timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <p>Powered by Stable Diffusion • Created with ❤️ for beautiful Kolam art</p>
      </footer>
    </div>
  );
}

export default App;
