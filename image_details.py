
import os
import numpy as np
from PIL import Image

# Dataset location
dataset_path = "dataset"

# Select one disease folder
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

# Convert image into NumPy array
img_array = np.array(img)

# Display image information
print("Image Size:", img.size)
print("Image Shape:", img_array.shape)
print("Pixel Values:", img_array[0][0])
print("Data Type:", img_array.dtype)
