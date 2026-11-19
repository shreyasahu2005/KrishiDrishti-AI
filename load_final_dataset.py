
import tensorflow as tf

# =====================================
# 1. DATASET CONFIGURATION
# =====================================

DATASET_DIR = "final_dataset"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42


# =====================================
# 2. LOAD TRAINING DATASET
# =====================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)


# =====================================
# 3. LOAD VALIDATION DATASET
# =====================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# =====================================
# 4. LOAD TEST DATASET
# =====================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# =====================================
# 5. DISPLAY CLASS NAMES
# =====================================

class_names = train_ds.class_names

print("\nClass names:")

for name in class_names:
    print(name)


# =====================================
# 6. MOBILE NET V2 PREPROCESSING
# =====================================

preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

train_ds = train_ds.map(
    lambda images, labels: (
        preprocess_input(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda images, labels: (
        preprocess_input(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda images, labels: (
        preprocess_input(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# =====================================
# 7. PERFORMANCE OPTIMIZATION
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)

test_ds = test_ds.prefetch(
    buffer_size=AUTOTUNE
)


# =====================================
# 8. VERIFY DATASET
# =====================================

print("\nChecking training dataset...")

for images, labels in train_ds.take(1):

    print("Image batch shape:", images.shape)

    print("Label batch shape:", labels.shape)

    print(
        "Minimum pixel value:",
        tf.reduce_min(images).numpy()
    )

    print(
        "Maximum pixel value:",
        tf.reduce_max(images).numpy()
    )


# =====================================
# 9. FINAL MESSAGE
# =====================================

print("\nAll datasets loaded successfully!")

print("MobileNetV2 preprocessing completed!")

print("Number of classes:", len(class_names))

print("Classes:", class_names)
