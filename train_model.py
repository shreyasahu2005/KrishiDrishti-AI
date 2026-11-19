
import tensorflow as tf
import numpy as np
import csv
from build_model import model
from load_final_dataset import (
    train_ds,
    val_ds,
    test_ds,
    class_names
)

# =====================================
# 1. TRAINING CONFIGURATION
# =====================================

EPOCHS = 10

MODEL_PATH = "krishidrishti_mobilenetv2.keras"


# =====================================
# 2. CALCULATE CLASS WEIGHTS
# =====================================

all_labels = np.concatenate([
    labels.numpy()
    for images, labels in train_ds
])

class_counts = np.bincount(
    all_labels,
    minlength=len(class_names)
)

class_weights = (
    len(all_labels) /
    (len(class_names) * class_counts)
)

class_weight_dict = {
    i: float(weight)
    for i, weight in enumerate(class_weights)
}

print("\nClass distribution:")

for i, name in enumerate(class_names):
    print(name, ":", class_counts[i])

print("\nClass weights:")
print(class_weight_dict)


# =====================================
# 3. CALLBACKS
# =====================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=0.000001
)


# =====================================
# 4. TRAIN THE MODEL
# =====================================

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)

# Save training history
with open("training_history.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(history.history.keys())

    writer.writerows(
        zip(*history.history.values())
    )

print("Training history saved successfully!")
# =====================================
# 5. SAVE FINAL MODEL
# =====================================

model.save(MODEL_PATH)

print("\nModel saved successfully!")


# =====================================
# 6. EVALUATE ON TEST DATA
# =====================================

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)

print("Test Accuracy:", test_accuracy)


# =====================================
# 7. FINAL RESULT
# =====================================

print("\nTraining completed!")

print("Model:", MODEL_PATH)

print("Classes:", class_names)

