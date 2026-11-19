
import tensorflow as tf
import numpy as np
import os
import time
import json

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. MODEL PATHS
# --------------------------------

original_path = "krishidrishti_mobilenetv2.keras"
finetuned_path = "krishidrishti_finetuned.keras"

# --------------------------------
# 2. LOAD TEST DATASET
# --------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    "final_dataset/test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:", class_names)

# MobileNetV2 preprocessing
def preprocess(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

test_ds = test_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Actual test labels
y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

# --------------------------------
# 3. MODEL EVALUATION FUNCTION
# --------------------------------

def evaluate_model(model_path, model_name):

    print("\n" + "=" * 50)
    print("Evaluating:", model_name)
    print("=" * 50)

    model = tf.keras.models.load_model(model_path)

    # Check model output
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output classes do not match dataset!")

    # Warm-up prediction
    model.predict(test_ds.take(1), verbose=0)

    # Measure prediction time
    start_time = time.perf_counter()

    y_prob = model.predict(test_ds, verbose=0)

    end_time = time.perf_counter()

    prediction_time = end_time - start_time

    # Predicted class
    y_pred = np.argmax(y_prob, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print("\nAccuracy:", accuracy)
    print("Macro F1 Score:", macro_f1)

    print(
        "Prediction time:",
        round(prediction_time, 3),
        "seconds"
    )

    print(
        "Time per image:",
        round((prediction_time / len(y_true)) * 1000, 3),
        "ms"
    )

    print("\nClassification Report:\n")

    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    print("\nConfusion Matrix:\n")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    print(cm)

    return {
        "name": model_name,
        "path": model_path,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "prediction_time": prediction_time
    }


# --------------------------------
# 4. EVALUATE BOTH MODELS
# --------------------------------

original_results = evaluate_model(
    original_path,
    "Original MobileNetV2"
)

finetuned_results = evaluate_model(
    finetuned_path,
    "Fine-Tuned MobileNetV2"
)

# --------------------------------
# 5. COMPARE MODELS
# --------------------------------

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print("\nOriginal Accuracy:",
      original_results["accuracy"])

print("Fine-Tuned Accuracy:",
      finetuned_results["accuracy"])

print("\nOriginal Macro F1:",
      original_results["macro_f1"])

print("Fine-Tuned Macro F1:",
      finetuned_results["macro_f1"])

# Macro F1 is the primary selection metric.
# Accuracy is used as a tie-breaker.

if (
    finetuned_results["macro_f1"],
    finetuned_results["accuracy"]
) > (
    original_results["macro_f1"],
    original_results["accuracy"]
):

    best_result = finetuned_results

else:

    best_result = original_results


# --------------------------------
# 6. SAVE BEST MODEL
# --------------------------------

print("\nSelected Model:", best_result["name"])

best_model = tf.keras.models.load_model(
    best_result["path"]
)

best_model.save("best_model.keras")

# Save class names for Streamlit
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("\nBest model saved successfully!")
print("File: best_model.keras")
print("Classes saved in: class_names.json")

