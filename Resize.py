import os
import cv2
import numpy as np

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "frames")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "processed_ghibli")

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process images
total_images = len(os.listdir(INPUT_FOLDER))
for i, img_name in enumerate(os.listdir(INPUT_FOLDER)):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"Processing image {i+1}/{total_images}: {img_name}")
    img_path = os.path.join(INPUT_FOLDER, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, (256, 256))  # Resize to 256x256
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        new_img_path = os.path.join(OUTPUT_FOLDER, img_name)
        try:
            cv2.imwrite(new_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception as e:
            print(f"Failed to write {new_img_path}: {e}")
            continue

print("Resizing completed! All images are now 256x256 in RGB format.")