
import os
import shutil
from sklearn.model_selection import train_test_split

# Original dataset path
SOURCE_DIR = "dataset"

# Final dataset path
OUTPUT_DIR = "final_dataset"

# Dataset categories
classes = os.listdir(SOURCE_DIR)

for cls in classes:

    class_path = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(class_path)

    # First split: 70% training, 30% temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # Second split: 15% validation, 15% testing
    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    # Create folders
    for split in ["train", "validation", "test"]:

        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )

    # Copy training images
    for img in train_images:

        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "train", cls, img)
        )

    # Copy validation images
    for img in val_images:

        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "validation", cls, img)
        )

    # Copy testing images
    for img in test_images:

        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "test", cls, img)
        )

    print(f"{cls} completed")

print("Dataset splitting completed successfully!")
