import cv2
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "to_split/temp"      # Folder with all your mixed kolam images
OUTPUT_DIR = "class2"      # Folder for the final, individual images

# --- TUNING PARAMETERS ---
# Use a moderate threshold; the script will invert it as needed.
THRESHOLD_VALUE = 150
# Connects small gaps in lines. Use (5, 5) if lines are still broken.
MORPH_KERNEL_SIZE = (3, 3)
# Filters out any small noise detected.
MIN_AREA_THRESHOLD = 500

# --- SCRIPT ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def is_background_light(image, corner_percent=5):
    """Checks if the image background is light by sampling corners."""
    try:
        height, width = image.shape
        corner_size = int(min(height, width) * (corner_percent / 100))
        corners = [
            image[0:corner_size, 0:corner_size],
            image[0:corner_size, width-corner_size:width],
            image[height-corner_size:height, 0:corner_size],
            image[height-corner_size:height, width-corner_size:width]
        ]
        avg_brightness = np.mean([np.mean(corner) for corner in corners])
        return avg_brightness > 127
    except Exception:
        return False  # Default to assuming a dark background on error


total_extracted = 0
print(f"Starting universal image processing from '{INPUT_DIR}'...")

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)

        original_img = cv2.imread(input_path)
        if original_img is None:
            print(f"⚠️  Warning: Could not read image {filename}. Skipping.")
            continue

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Determine the correct thresholding method based on background
        if is_background_light(gray_img):
            # For dark lines on a light background
            threshold_type = cv2.THRESH_BINARY_INV
        else:
            # For light lines on a dark background
            threshold_type = cv2.THRESH_BINARY

        _, binary_img = cv2.threshold(
            gray_img, THRESHOLD_VALUE, 255, threshold_type)

        # Perform Morphological Closing to connect broken lines and dots
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        # Find components on the clean, 'closed' image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed_img, 4, cv2.CV_32S)

        extracted_count_per_file = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= MIN_AREA_THRESHOLD:
                x, y, w, h = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT],
                )

                cropped_kolam = original_img[y:y+h, x:x+w]

                base_filename = os.path.splitext(filename)[0]
                output_path = os.path.join(
                    OUTPUT_DIR, f"{base_filename}_part_{i}.png")
                cv2.imwrite(output_path, cropped_kolam)
                extracted_count_per_file += 1

        if extracted_count_per_file > 0:
            total_extracted += extracted_count_per_file
            print(
                f"Processed '{filename}': Extracted {extracted_count_per_file} Kolam(s).")

print(
    f"\n✅ Finished! Extracted a total of {total_extracted} individual Kolams into '{OUTPUT_DIR}'.")
