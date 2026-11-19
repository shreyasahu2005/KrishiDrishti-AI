
from PIL import Image
from pathlib import Path

from utils import predict_disease


# Find one test image
test_folder = Path("final_dataset/test")

image_paths = [
    p for p in test_folder.rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

# Select first image
image_path = image_paths[0]

# Open image
image = Image.open(image_path)

# Predict disease
predicted_class, confidence = predict_disease(image)

# Actual class from folder name
actual_class = image_path.parent.name

# Display results
print("\nActual Disease:", actual_class)

print("Predicted Disease:", predicted_class)

print("Confidence:", round(confidence, 2), "%")

print("\nPrediction testing completed!")
