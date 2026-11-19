
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "dataset"

# Select Tomato Early Blight folder
folder_path = os.path.join(
    dataset_path,
    "Tomato___Early_blight"
)

# Get image names
images = os.listdir(folder_path)

# Select first image
image_path = os.path.join(
    folder_path,
    images[0]
)

# Open image
img = Image.open(image_path)

# Resize image
resized_img = img.resize((224, 224))

# Convert image into NumPy array
img_array = np.array(resized_img)

# Normalize image
normalized_img = img_array / 255.0

# Print information
print("Original Size:", img.size)
print("Resized Shape:", img_array.shape)
print("Normalized Shape:", normalized_img.shape)

print("Original Pixel:", img_array[0][0])
print("Normalized Pixel:", normalized_img[0][0])

print("Minimum Value:", normalized_img.min())
print("Maximum Value:", normalized_img.max())

# Display resized image
plt.imshow(normalized_img)
plt.title("Resized Tomato Leaf")
plt.axis("off")
plt.show()

