
import tensorflow as tf

from load_final_dataset import (
    train_ds,
    val_ds,
    test_ds,
    class_names
)

# =====================================
# 1. LOAD PRETRAINED MOBILENETV2
# =====================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# =====================================
# 2. FREEZE PRETRAINED LAYERS
# =====================================

base_model.trainable = False


# =====================================
# 3. CREATE INPUT LAYER
# =====================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)


# =====================================
# 4. FEATURE EXTRACTION
# =====================================

x = base_model(
    inputs,
    training=False
)


# =====================================
# 5. GLOBAL AVERAGE POOLING
# =====================================

x = tf.keras.layers.GlobalAveragePooling2D()(x)


# =====================================
# 6. DROPOUT LAYER
# =====================================

x = tf.keras.layers.Dropout(0.3)(x)


# =====================================
# 7. OUTPUT CLASSIFICATION LAYER
# =====================================

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)


# =====================================
# 8. BUILD FINAL MODEL
# =====================================

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# =====================================
# 9. COMPILE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================
# 10. DISPLAY MODEL SUMMARY
# =====================================

model.summary()

print("\nNumber of classes:", len(class_names))
print("Classes:", class_names)

print("\nMobileNetV2 model built successfully!")
