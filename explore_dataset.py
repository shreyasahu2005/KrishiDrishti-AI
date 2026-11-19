
import os

dataset_path = "dataset"

classes = os.listdir(dataset_path)

print("Total Classes:", len(classes))

print("\nDisease Classes:")

for class_name in classes:
    print(class_name)
