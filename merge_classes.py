import os
import shutil

# Folders
splits = ["train", "val", "test"]

# Mapping old folder names → new class names
rename_map = {
    "Augmented Banana Black Sigatoka Disease": "black_sigatoka",
    "Augmented Banana Moko Disease": "moko_disease",
    "Augmented Banana Panama Disease": "panama_disease",
    "Augmented Banana Yellow Sigatoka Disease": "yellow_sigatoka",
    "Augmented Banana Healthy Leaf_dup": "healthy"
}

for split in splits:
    split_path = os.path.join(split)
    for old_name, new_name in rename_map.items():
        old_path = os.path.join(split_path, old_name)
        new_path = os.path.join(split_path, new_name)
        if os.path.exists(old_path):
            os.makedirs(new_path, exist_ok=True)
            # Move all images into new folder
            for img in os.listdir(old_path):
                shutil.move(os.path.join(old_path, img), os.path.join(new_path, img))
            # Remove old empty folder
            os.rmdir(old_path)

print("Classes merged successfully!")