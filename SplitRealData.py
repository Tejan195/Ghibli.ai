import random
import os
from shutil import copyfile

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset", "processed_normal")
train_path = os.path.join(BASE_DIR, "dataset", "train_x")
test_path = os.path.join(BASE_DIR, "dataset", "test_x")

# Create directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Filter for image files
images = [img for img in os.listdir(dataset_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle with a seed for reproducibility
random.seed(42)
random.shuffle(images)

# Split into train/test (80/20)
train_ratio = 0.8
split_idx = int(len(images) * train_ratio)
train_images, test_images = images[:split_idx], images[split_idx:]

# Copy training images
for i, img in enumerate(train_images):
    print(f"Copying train image {i+1}/{len(train_images)}: {img}")
    try:
        copyfile(os.path.join(dataset_path, img), os.path.join(train_path, img))
    except Exception as e:
        print(f"Failed to copy {img} to train: {e}")
        continue

# Copy test images
for i, img in enumerate(test_images):
    print(f"Copying test image {i+1}/{len(test_images)}: {img}")
    try:
        copyfile(os.path.join(dataset_path, img), os.path.join(test_path, img))
    except Exception as e:
        print(f"Failed to copy {img} to test: {e}")
        continue

print(f"Train: {len(train_images)}, Test: {len(test_images)}")