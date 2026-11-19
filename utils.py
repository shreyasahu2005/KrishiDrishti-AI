
import tensorflow as tf
import numpy as np

from PIL import Image


# Load trained model
model = tf.keras.models.load_model("best_model.keras")


# Load class names
import json

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# Image preprocessing function
def preprocess_image(image):

    # Convert image into RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image into NumPy array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return image_array


# Prediction function
def predict_disease(image):

    # Preprocess uploaded image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)

    # Get predicted class index
    predicted_index = np.argmax(predictions[0])

    # Get disease name
    predicted_class = class_names[predicted_index]

    # Get confidence
    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_class, confidence
