import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import json

# ==========================
# CONFIGURATION
# ==========================
dataset_path = "dataset"
img_height = 224
img_width = 224
batch_size = 16   # smaller batch size for better learning

# ==========================
# LOAD DATASET
# ==========================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Class Names:", class_names)

# Save class names for Flask app
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==========================
# DATA AUGMENTATION
# ==========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================
# PRETRAINED MODEL
# ==========================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze pretrained layers

# ==========================
# BUILD MODEL
# ==========================
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# ==========================
# COMPILE MODEL
# ==========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# TRAIN MODEL
# ==========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12
)

# ==========================
# SAVE MODEL
# ==========================
model.save("crop_disease_model.h5")

# ==========================
# PLOT ACCURACY GRAPH
# ==========================
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_graph.png")
plt.show()

print("Model Training Completed Successfully!")