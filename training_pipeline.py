
import tensorflow as tf

# Dataset path
dataset_path = "dataset"

# Constants
IMAGE_SIZE = (224, 224)
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

# Save class names
class_names = train_dataset.class_names

print("Classes:", class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply augmentation and normalization to training data
train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y)
)

# Apply only normalization to validation data
validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

print("Training pipeline ready!")

# Check one batch
for images, labels in train_dataset.take(1):
    print("Training Image Shape:", images.shape)
    print("Training Label Shape:", labels.shape)

for images, labels in validation_dataset.take(1):
    print("Validation Image Shape:", images.shape)
    print("Validation Label Shape:", labels.shape)
