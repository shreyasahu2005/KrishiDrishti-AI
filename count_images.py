
import os

dataset_path = "dataset"

classes = os.listdir(dataset_path)

for class_name in classes:

    folder_path = os.path.join(dataset_path, class_name)

    images = os.listdir(folder_path)

    image_count = len(images)

    print(class_name, ":", image_count)
