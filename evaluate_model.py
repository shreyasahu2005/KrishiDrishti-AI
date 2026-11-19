
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from load_final_dataset import test_ds, class_names


# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

model = tf.keras.models.load_model(
    "krishidrishti_mobilenetv2.keras"
)

print("Trained model loaded successfully!")


# =====================================
# 2. LOAD TRAINING HISTORY
# =====================================

history = pd.read_csv("training_history.csv")


# =====================================
# 3. ACCURACY GRAPH
# =====================================

plt.figure(figsize=(8, 5))

plt.plot(history["accuracy"], label="Training Accuracy")

plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig("accuracy_graph.png")

plt.show()


# =====================================
# 4. LOSS GRAPH
# =====================================

plt.figure(figsize=(8, 5))

plt.plot(history["loss"], label="Training Loss")

plt.plot(history["val_loss"], label="Validation Loss")

plt.title("Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig("loss_graph.png")

plt.show()


# =====================================
# 5. PREDICT TEST DATA
# =====================================

y_prob = model.predict(test_ds)

y_pred = np.argmax(y_prob, axis=1)


# =====================================
# 6. GET ACTUAL LABELS
# =====================================

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])


# =====================================
# 7. CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# =====================================
# 8. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("KrishiDrishti AI - Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


# =====================================
# 9. FINAL TEST ACCURACY
# =====================================

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nFinal Test Results")

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy * 100, "%")


print("\nModel evaluation completed!")

