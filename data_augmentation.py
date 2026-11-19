
import tensorflow as tf
import matplotlib.pyplot as plt
from load_dataset import train_dataset
# Create data augmentation layer
data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1),

])

# Load one image from training dataset
for images, labels in train_dataset.take(1):
    original_image = images[0]

# Apply augmentation
augmented_image = data_augmentation(
    tf.expand_dims(original_image, axis=0),
    training=True
)

# Display original and augmented images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image[0])
plt.title("Augmented Image")
plt.axis("off")

plt.show()
