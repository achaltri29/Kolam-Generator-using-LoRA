from PIL import Image
import os

input_folder = "yes"
output_folder = "temp_preprocess"
size = (128, 128)

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(input_folder, file)).convert("L")  # grayscale
        img = img.resize(size)  # resize
        img.save(os.path.join(output_folder, file))
