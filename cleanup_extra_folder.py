import os
import shutil

splits = ["train", "val", "test"]
extra_folder = "Banana Disease Recognition Dataset"

for split in splits:
    path = os.path.join(split, extra_folder)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed extra folder: {path}")

print("Cleanup completed!")