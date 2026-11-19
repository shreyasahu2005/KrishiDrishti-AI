
import os

for split in ["train", "validation", "test"]:

    print("\n", split.upper())

    split_path = os.path.join("final_dataset", split)

    total = 0

    for cls in os.listdir(split_path):

        class_path = os.path.join(split_path, cls)

        count = len(os.listdir(class_path))

        total += count

        print(cls, ":", count)

    print("Total images:", total)
