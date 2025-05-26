import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = Path("hand_gestures/images")
train_dir = Path("data/train")
valid_dir = Path("data/valid")

# Make sure destination folders exist
train_dir.mkdir(parents=True, exist_ok=True)
valid_dir.mkdir(parents=True, exist_ok=True)

# Fix the random seed for reproducibility
random.seed(42)

# Iterate over each category/class folder
for class_folder in source_dir.iterdir():
    if class_folder.is_dir():
        class_name = class_folder.name
        images = list(class_folder.glob("*.*"))  # You can add filters if needed (e.g. "*.jpg")

        # Shuffle and split
        random.shuffle(images)
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        valid_images = images[split_idx:]

        # Create corresponding subfolders in train/valid
        train_class_dir = train_dir / class_name
        valid_class_dir = valid_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        valid_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for img_path in train_images:
            shutil.copy(img_path, train_class_dir / img_path.name)

        for img_path in valid_images:
            shutil.copy(img_path, valid_class_dir / img_path.name)

        print(f"Class '{class_name}': {len(train_images)} train, {len(valid_images)} valid")

print("\n✅ Dataset split completed.")
