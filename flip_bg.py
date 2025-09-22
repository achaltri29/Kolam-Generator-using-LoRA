import os

# --- Configuration ---
# Make sure this matches the folder name in your dcgan.py script
DATA_FOLDER = "final_shayad/class1/"

# --- Script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, DATA_FOLDER)

print(f"Checking for dataset at path: {data_path}\n")

if not os.path.isdir(data_path):
    print(f"❌ ERROR: The folder '{DATA_FOLDER}' was not found.")
    print("Please make sure the folder exists and the name is correct.")
else:
    print(f"✅ SUCCESS: Folder '{DATA_FOLDER}' found.")

    image_files = [f for f in os.listdir(
        data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) == 0:
        print(f"❌ ERROR: The folder exists, but it contains 0 valid image files.")
        print("Please run your preprocessing scripts again to populate this folder.")
    else:
        print(f"✅ SUCCESS: Found {len(image_files)} valid image files.")
        print("Your dataset seems to be ready. Try running the DCGAN script again.")
