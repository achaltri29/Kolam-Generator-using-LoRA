import cv2
import os
import numpy as np

# --- CONFIGURATION ---
# The folder with your preprocessed images that might have borders
INPUT_DIR = "inverted"

# The final, tightly cropped dataset will be saved here
OUTPUT_DIR = "white_padding_removed"

# --- TUNING PARAMETERS ---
# How close to pure white a pixel must be to be considered part of the border. (0-255)
# A high value like 250 is good for clean white borders.
THRESHOLD_VALUE = 250

# How many pixels of padding to add around the cropped content.
PADDING_PX = 10

# --- SCRIPT ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cropped_count = 0
skipped_count = 0
print(f"Starting to auto-crop borders from images in '{INPUT_DIR}'...")

all_files = [f for f in os.listdir(
    INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for i, filename in enumerate(all_files):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Processing image {i+1}/{len(all_files)}: {filename}", end='\r')

    try:
        # Load the image in grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped_count += 1
            continue

        # 1. Create a binary mask of the content
        # We use an inverted threshold: it turns the white border black and the content white.
        _, thresh = cv2.threshold(
            img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)

        # 2. Find the coordinates of all content pixels
        coords = cv2.findNonZero(thresh)
        if coords is None:
            # This means the image was completely white or empty
            skipped_count += 1
            continue

        # 3. Get the bounding box that encloses all content
        x, y, w, h = cv2.boundingRect(coords)

        # 4. Add padding (but don't go outside the original image dimensions)
        x_start = max(x - PADDING_PX, 0)
        y_start = max(y - PADDING_PX, 0)
        x_end = min(x + w + PADDING_PX, img.shape[1])
        y_end = min(y + h + PADDING_PX, img.shape[0])

        # 5. Crop the original image using the padded coordinates
        cropped_image = img[y_start:y_end, x_start:x_end]

        # Save the final cropped image
        cv2.imwrite(output_path, cropped_image)
        cropped_count += 1

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        skipped_count += 1

print("\n" + "="*50)
print("✅ Auto-cropping complete!")
print(f"   Successfully cropped: {cropped_count} images.")
print(f"   Skipped (e.g., empty images): {skipped_count} images.")
print(f"Your final, tightly cropped dataset is in '{OUTPUT_DIR}'.")
