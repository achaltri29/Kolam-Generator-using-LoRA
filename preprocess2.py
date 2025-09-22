import cv2
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "temp_preprocess"            # Folder with your 1700+ single kolam images
OUTPUT_DIR = "processed_256x256"  # Final, DCGAN-ready dataset will be saved here

# --- PROCESSING PARAMETERS ---
# The final dimension of the output images (256x256)
TARGET_SIZE = 256
# Helps separate the lines from the background.
THRESHOLD_VALUE = 100
# Ignores any leftover noise smaller than this.
MIN_CONTOUR_AREA = 500

# --- SCRIPT ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def process_image(input_path, output_path):
    """
    Extracts the main Kolam feature, centers it on a square canvas,
    and resizes it to the target dimension.
    """
    # 1. Load the image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # 2. Isolate the Kolam using a binary threshold
    # This creates a black-and-white mask of the drawing
    _, binary_img = cv2.threshold(img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # 3. Find the largest contour (the Kolam itself)
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False  # Skip if no contours are found

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < MIN_CONTOUR_AREA:
        return False  # Skip if the largest object is just noise

    # 4. Get the bounding box of the main Kolam
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped_kolam = binary_img[y:y+h, x:x+w]

    # 5. Create a square canvas and center the Kolam to preserve its aspect ratio
    # Determine the size of the new square canvas
    canvas_size = max(w, h)
    # Create a black square canvas
    square_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Calculate coordinates to paste the cropped kolam in the center
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2

    # Paste the kolam onto the canvas
    square_canvas[paste_y:paste_y+h, paste_x:paste_x+w] = cropped_kolam

    # 6. Resize the final square canvas to the target size (e.g., 256x256)
    # Using INTER_AREA is best for downscaling to avoid artifacts
    final_image = cv2.resize(
        square_canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # 7. Save the processed image
    cv2.imwrite(output_path, final_image)
    return True


# --- MAIN EXECUTION ---
processed_count = 0
skipped_count = 0
print(f"Starting dataset processing from '{INPUT_DIR}'...")

all_files = os.listdir(INPUT_DIR)
for i, filename in enumerate(all_files):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Print progress
        print(f"Processing image {i+1}/{len(all_files)}: {filename}", end='\r')

        try:
            if process_image(input_path, output_path):
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped_count += 1

print("\n" + "="*50)
print("✅ Processing complete!")
print(f"   Successfully processed and saved: {processed_count} images.")
print(f"   Skipped (no valid Kolam found): {skipped_count} images.")
print(f"Your DCGAN-ready dataset is in the '{OUTPUT_DIR}' folder.")
