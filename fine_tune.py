
import tensorflow as tf

from load_final_dataset import train_ds, val_ds

# =====================================
# 1. LOAD PREVIOUSLY TRAINED MODEL
# =====================================

MODEL_PATH = "krishidrishti_mobilenetv2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Previously trained model loaded!")


# =====================================
# 2. FIND MOBILENETV2 BASE MODEL
# =====================================

base_model = model.get_layer("mobilenetv2_1.00_224")

print("MobileNetV2 base model found!")


# =====================================
# 3. UNFREEZE BASE MODEL
# =====================================

base_model.trainable = True


# =====================================
# 4. FREEZE EARLIER LAYERS
# =====================================

for layer in base_model.layers[:-30]:

    layer.trainable = False


# =====================================
# 5. KEEP BATCH NORMALIZATION FROZEN
# =====================================

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# =====================================
# 6. COMPILE WITH LOW LEARNING RATE
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================
# 7. DISPLAY TRAINABLE PARAMETERS
# =====================================

print("\nTrainable layers:")

for layer in base_model.layers:

    if layer.trainable:

        print(layer.name)


# =====================================
# 8. FINE-TUNE MODEL
# =====================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "krishidrishti_finetuned.keras",
    monitor="val_loss",
    save_best_only=True
)

print("\nStarting fine-tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# =====================================
# 9. SAVE FINAL MODEL
# =====================================

model.save("krishidrishti_finetuned.keras")

print("\nFine-tuning completed!")

print("Fine-tuned model saved successfully!")
