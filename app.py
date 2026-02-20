from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
import base64
import os
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Kolam AI Generator", description="Beautiful AI-generated Kolam patterns")

# Allow all origins for simplicity, you might want to restrict this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

class Prompt(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, deformed, text, watermark, ugly"
    steps: Optional[int] = 50
    width: Optional[int] = 512
    height: Optional[int] = 512
    cfg_scale: Optional[float] = 7.0
    sampler_index: Optional[str] = "Euler"

@app.post("/generate")
async def generate_image(prompt_data: Prompt):
    try:
        url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        base_prompt = "a highly detailed SKS kolam, intricate symmetrical pattern, masterpiece, sharp focus"
        
        # Combine base prompt with user prompt
        full_prompt = f"{base_prompt}, {prompt_data.prompt}" if prompt_data.prompt.strip() else base_prompt
        
        payload = {
            "prompt": full_prompt,
            "negative_prompt": prompt_data.negative_prompt,
            "steps": prompt_data.steps,
            "width": prompt_data.width,
            "height": prompt_data.height,
            "cfg_scale": prompt_data.cfg_scale,
            "sampler_index": prompt_data.sampler_index,
            "send_images": True,
            "save_images": False
        }

        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate image from Stable Diffusion API")
            
        data = response.json()
        
        if not data.get("images") or len(data["images"]) == 0:
            raise HTTPException(status_code=500, detail="No image generated")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"kolam_{timestamp}_{unique_id}.png"
        filepath = os.path.join("output", filename)
        
        # Save image locally
        image_base64 = data["images"][0]
        image_bytes = base64.b64decode(image_base64)
        
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        # --- ENHANCE IMAGE AUTOMATICALLY ---
        try:
            from enhance_kolam import enforce_symmetry
            print(f"Enhancing image: {filepath}")
            # Overwrite the original file with the enhanced version
            enforce_symmetry(filepath, filepath)
            
            # Reload the enhanced image to return correct base64 to frontend
            with open(filepath, "rb") as f:
                enhanced_bytes = f.read()
                image_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')
                
        except Exception as e:
            print(f"Warning: Auto-enhancement failed: {e}")
            # Continue with original image if enhancement fails
        # -----------------------------------
        
        return {
            "image": image_base64,
            "filename": filename,
            "prompt": full_prompt,
            "timestamp": timestamp
        }
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Stable Diffusion WebUI is not running. Please start it with --api flag.")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timed out. The image generation is taking too long.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.get("/images/{filename}")
async def get_image(filename: str):
    filepath = os.path.join("output", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Kolam AI Generator is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
