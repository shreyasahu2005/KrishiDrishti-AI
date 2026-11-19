
import tensorflow as tf

# Dataset location
dataset_path = "dataset"

# Image size
IMAGE_SIZE = (224, 224)

# Batch size
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Get class names
class_names = train_dataset.class_names

print("Disease Classes:", class_names)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# Display dataset information
for images, labels in train_dataset.take(1):

    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)

    print("Minimum Pixel Value:", tf.reduce_min(images).numpy())
    print("Maximum Pixel Value:", tf.reduce_max(images).numpy())

print("Dataset preprocessing completed!")
