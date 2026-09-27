import os
import random
import shutil

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Paths
dataset_dir = "dataset"
augmented_dir = os.path.join(dataset_dir, "Banana Disease Recognition Dataset", "Augmented images")
splits_dirs = ["train", "val", "test"]

# Create train/val/test folders if not exist
for folder in splits_dirs:
    os.makedirs(folder, exist_ok=True)

# Collect all class folders
class_folders = []

# Original dataset classes
for f in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, f)
    if os.path.isdir(path) and f.lower() not in ["banana disease recognition dataset"]:
        class_folders.append(path)

# Augmented dataset classes
if os.path.exists(augmented_dir):
    for f in os.listdir(augmented_dir):
        path = os.path.join(augmented_dir, f)
        if os.path.isdir(path):
            # Include nested folders
            for subf in os.listdir(path):
                sub_path = os.path.join(path, subf)
                if os.path.isdir(sub_path):
                    class_folders.append(sub_path)

# Dictionary to store summary
summary = {"train": {}, "val": {}, "test": {}}

# Split images
for class_path in class_folders:
    class_name = os.path.basename(class_path)
    images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    for idx, img in enumerate(images):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        dest_folder = os.path.join(split, class_name)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(img, os.path.join(dest_folder, os.path.basename(img)))

        # Update summary
        summary[split][class_name] = summary[split].get(class_name, 0) + 1

# Print summary
print("Dataset split completed successfully!\n")
for split in splits_dirs:
    print(f"{split.upper()}:")
    for class_name, count in summary[split].items():
        print(f"  {class_name}: {count} images")
    print()