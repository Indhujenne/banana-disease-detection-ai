import os

splits = ["train", "val", "test"]

print("✅ Final Dataset Summary:\n")

for split in splits:
    split_path = os.path.join(split)
    print(f"{split.upper()}:")
    if not os.path.exists(split_path):
        print("  Folder does not exist!")
        continue
    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {class_name}: {count} images")
    print()