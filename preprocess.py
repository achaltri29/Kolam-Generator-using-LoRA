import cv2
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "temp_preprocess"  # Your clean, standardized dataset
OUTPUT_DIR = "grayscaled"      # The final folder for training

# --- PROCESSING PARAMETERS ---
TARGET_SIZE = 256
# Threshold to make the image pure black and white after processing
FINAL_THRESHOLD_VALUE = 128

# --- SCRIPT ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

processed_count = 0
skipped_count = 0
print(f"Starting final preprocessing from '{INPUT_DIR}'...")

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

        # 1. ## Center and Pad to Preserve Aspect Ratio ##
        # Find the bounding box of the white Kolam lines
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            skipped_count += 1
            continue

        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        # Crop the Kolam precisely
        cropped_kolam = img[y:y+h, x:x+w]

        # Create a black square canvas with the size of the largest dimension
        canvas_size = max(w, h)
        square_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # Calculate coordinates to paste the cropped kolam in the center
        paste_x = (canvas_size - w) // 2
        paste_y = (canvas_size - h) // 2

        # Paste the kolam onto the canvas
        square_canvas[paste_y:paste_y+h, paste_x:paste_x+w] = cropped_kolam

        # 2. ## Resize to 256x256 ##
        # Now, resize the perfectly square canvas to the target size
        resized_image = cv2.resize(
            square_canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # 3. ## Binarize to ensure crisp Black and White ##
        # This removes any gray pixels introduced by resizing
        _, final_image = cv2.threshold(
            resized_image, FINAL_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # Save the final, DCGAN-ready image
        cv2.imwrite(output_path, final_image)
        processed_count += 1

    except Exception as e:
        print(f"\nError processing {filename}: {e}")
        skipped_count += 1

print("\n" + "="*50)
print("✅ Final preprocessing complete!")
print(f"   Successfully processed: {processed_count} images.")
print(f"   Skipped or failed: {skipped_count} images.")
print(f"Your final, DCGAN-ready dataset is in '{OUTPUT_DIR}'.")
