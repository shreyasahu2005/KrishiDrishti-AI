
import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "dataset"

folder_path = os.path.join(
    dataset_path,
    "Tomato___Early_blight"
)

images = os.listdir(folder_path)

image_path = os.path.join(
    folder_path,
    images[0]
)

img = Image.open(image_path)

plt.imshow(img)
plt.title("Tomato Early Blight")
plt.axis("off")
plt.show()
