
import tensorflow as tf
import numpy as np
import json
import os
from pathlib import Path

# Load best model
model = tf.keras.models.load_model("best_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of classes:", len(class_names))

# Find one test image
test_path = Path("final_dataset/test")

image_paths = [
    p for p in test_path.rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

if not image_paths:
    raise ValueError("No test images found!")

image_path = image_paths[0]

# Load image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Apply MobileNetV2 preprocessing
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    img_array
)

# Predict
predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])

print("\nImage:", image_path)
print("Predicted disease:", class_names[predicted_index])
print("Confidence:", round(float(predictions[0][predicted_index]) * 100, 2), "%")

print("\nModel verification completed!")
