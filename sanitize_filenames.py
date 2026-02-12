import os
import re

# --- CONFIGURATION ---
# Point this to the folder with the messy filenames
TARGET_DIR = "to_split/temp"

# --- SCRIPT ---
print(f"Scanning for files to rename in '{TARGET_DIR}'...")
renamed_count = 0

for filename in os.listdir(TARGET_DIR):
    # Replace any character that is not a letter, number, dot, underscore, or hyphen with an underscore
    safe_filename = re.sub(r'[^\w\.\-]', '_', filename)

    if filename != safe_filename:
        original_path = os.path.join(TARGET_DIR, filename)
        new_path = os.path.join(TARGET_DIR, safe_filename)

        try:
            os.rename(original_path, new_path)
            print(f"Renamed: '{filename}' -> '{safe_filename}'")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming '{filename}': {e}")

print(f"\n✅ Done. Renamed {renamed_count} files.")
