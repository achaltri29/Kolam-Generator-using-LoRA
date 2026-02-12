import requests
import base64
from io import BytesIO
from PIL import Image

# API endpoint
url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

# Minimal payload with correct values
payload = {
    "prompt": "a highly detailed SKS kolam, intricate symmetrical pattern, masterpiece, sharp focus",
    "negative_prompt": "blurry, deformed, text, watermark, ugly",
    "steps": 50,
    "width": 512,
    "height": 512,
    "cfg_scale": 7,
    "sampler_index": "Euler",
    "send_images": True,
    "save_images": False  # no need to save via WebUI, we handle it here
}

# Send request
response = requests.post(url, json=payload)
data = response.json()

# Decode returned image
image_base64 = data["images"][0]
image_bytes = base64.b64decode(image_base64)

# Save locally
image_path = "kolam.png"
with open(image_path, "wb") as f:
    f.write(image_bytes)

# Open with PIL
image = Image.open(BytesIO(image_bytes))
image.show()

print(f"Image saved as {image_path}")
