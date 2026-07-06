"""
=========================================================
Day 16 - Transfer Learning with MobileNetV2
Part 1 - Dataset Loading & Visualization
=========================================================
"""

# ==========================
# Import Libraries
# ==========================

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# Configuration
# ==========================

DATASET_PATH = "dataset/flowers"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ==========================
# Load Training Dataset
# ==========================

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ==========================
# Load Validation Dataset
# ==========================

validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ==========================
# Get Class Names
# ==========================

class_names = train_dataset.class_names

print("\n==============================")
print("Flower Classes")
print("==============================")

for i, name in enumerate(class_names):
    print(f"{i} --> {name}")

# ==========================
# Preprocess Images
# ==========================

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

# ==========================
# Improve Performance
# ==========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(AUTOTUNE)

validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)

# ==========================
# Display Sample Images
# ==========================

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):

    # Reverse preprocessing ONLY for display
    images = (images + 1) / 2

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy())

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.tight_layout()
plt.show()

print("\nDataset Loaded Successfully.")
print("Ready for Transfer Learning.")

"""
=========================================================
Part 2 - Feature Extraction using MobileNetV2
=========================================================
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ==========================
# Load Pretrained MobileNetV2
# ==========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ==========================
# Freeze Base Model
# ==========================

base_model.trainable = False

print("\n=================================")
print("Base Model Frozen")
print("=================================")
print("Trainable Layers:", len(base_model.trainable_variables))

# ==========================
# Build Custom Classifier
# ==========================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)

x = Dropout(0.3)(x)

predictions = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Model Summary
# ==========================

print("\n=================================")
print("Model Summary")
print("=================================")

model.summary()

# ==========================
# Train Model
# ==========================

print("\n=================================")
print("Training Started")
print("=================================")

history_feature = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# ==========================
# Create Folders
# ==========================

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ==========================
# Save Model
# ==========================

model.save("models/feature_extraction.keras")

print("\nFeature Extraction Model Saved.")

# ==========================
# Plot Accuracy
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history_feature.history["accuracy"], label="Training Accuracy")
plt.plot(history_feature.history["val_accuracy"], label="Validation Accuracy")

plt.title("Feature Extraction Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("plots/feature_extraction_accuracy.png")
plt.show()

# ==========================
# Plot Loss
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history_feature.history["loss"], label="Training Loss")
plt.plot(history_feature.history["val_loss"], label="Validation Loss")

plt.title("Feature Extraction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("plots/feature_extraction_loss.png")
plt.show()

print("\n=================================")
print("Feature Extraction Completed")
print("=================================")

"""
=========================================================
Part 3 - Fine Tuning MobileNetV2
=========================================================
"""

# ==========================
# Unfreeze Base Model
# ==========================

base_model.trainable = True

# Freeze all layers except last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

print("\n=================================")
print("Fine-Tuning Setup")
print("=================================")

trainable_count = sum(layer.trainable for layer in base_model.layers)

print(f"Total Layers       : {len(base_model.layers)}")
print(f"Trainable Layers   : {trainable_count}")
print(f"Frozen Layers      : {len(base_model.layers)-trainable_count}")

# ==========================
# Recompile Model
# ==========================

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=================================")
print("Fine-Tuning Started")
print("=================================")

# ==========================
# Train Again
# ==========================

history_finetune = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# ==========================
# Save Final Model
# ==========================

model.save("models/fine_tuned_model.keras")

print("\nFine-Tuned Model Saved.")

# ==========================
# Accuracy Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history_finetune.history["accuracy"], label="Training Accuracy")
plt.plot(history_finetune.history["val_accuracy"], label="Validation Accuracy")

plt.title("Fine-Tuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("plots/fine_tuning_accuracy.png")
plt.show()

# ==========================
# Loss Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history_finetune.history["loss"], label="Training Loss")
plt.plot(history_finetune.history["val_loss"], label="Validation Loss")

plt.title("Fine-Tuning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("plots/fine_tuning_loss.png")
plt.show()

# ==========================
# Compare Results
# ==========================

feature_acc = history_feature.history["val_accuracy"][-1]
fine_acc = history_finetune.history["val_accuracy"][-1]

print("\n=================================")
print("Comparison")
print("=================================")
print(f"Feature Extraction Validation Accuracy : {feature_acc:.4f}")
print(f"Fine-Tuning Validation Accuracy         : {fine_acc:.4f}")

if fine_acc > feature_acc:
    print("\nFine-Tuning Improved the Model.")
elif fine_acc < feature_acc:
    print("\nFeature Extraction Performed Better.")
else:
    print("\nBoth methods achieved the same accuracy.")