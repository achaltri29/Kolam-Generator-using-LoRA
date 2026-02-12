import cv2
import os

# --- CONFIGURATION ---
# 1. Place all the images you want to invert into this folder
INPUT_DIR = "manual_invert/temp"

# 2. The inverted images will be saved here
OUTPUT_DIR = "manual_inverted/temp"

# --- SCRIPT ---
# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
    print(
        f"⚠️  Warning: Input folder '{INPUT_DIR}' did not exist. I've created it for you.")
    print("Please place the images you want to invert inside it and run the script again.")
    exit()

inverted_count = 0
print(f"Starting inversion process for folder '{INPUT_DIR}'...")

# Loop through all files in the input directory
for filename in os.listdir(INPUT_DIR):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Read the image in grayscale
        gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Skip if the image could not be read
        if gray_img is None:
            print(f"⚠️  Warning: Could not read {filename}. Skipping.")
            continue

        # ## CORE LOGIC: Invert the colors ##
        # cv2.bitwise_not flips every black pixel to white and every white pixel to black.
        inverted_image = cv2.bitwise_not(gray_img)

        # Save the new, inverted image
        cv2.imwrite(output_path, inverted_image)
        inverted_count += 1

print("\n" + "="*50)
print("✅ Inversion complete!")
print(f"   Successfully inverted and saved {inverted_count} images.")
print(f"Your inverted images are in the '{OUTPUT_DIR}' folder.")
