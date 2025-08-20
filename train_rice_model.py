# train_rice_model.py
# This file is functionally correct and requires no changes.

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from sklearn.model_selection import train_test_split

# Set directories
data_dir = "sample_images/Rice_Diseases"  # Adjust to your local path
train_dir = "train"
val_dir = "val"

# Check if data directory exists and contains subdirectories
print(f"Checking directory: {data_dir}")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found. Please create it and add image subdirectories.")
print(f"Subdirectories: {os.listdir(data_dir)}")

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List of disease classes (should match subdirectory names)
diseases = ['Bacterial Blight Disease', 'Blast Disease', 'Brown Spot Disease', 'False Smut Disease']

# Split images into train and validation sets
for disease in diseases:
    disease_dir = os.path.join(data_dir, disease)
    if not os.path.exists(disease_dir):
        raise FileNotFoundError(f"Subdirectory {disease_dir} not found. Please add images.")
    images = os.listdir(disease_dir)
    print(f"Found {len(images)} images in {disease}")
    if not images:
        raise ValueError(f"No images found in {disease_dir}")
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    train_disease_dir = os.path.join(train_dir, disease)
    val_disease_dir = os.path.join(val_dir, disease)
    os.makedirs(train_disease_dir, exist_ok=True)
    os.makedirs(val_disease_dir, exist_ok=True)

    for image in train_images:
        shutil.copy(os.path.join(disease_dir, image), os.path.join(train_disease_dir, image))
    for image in val_images:
        shutil.copy(os.path.join(disease_dir, image), os.path.join(val_disease_dir, image))

# Data loading and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Train dataset size: {train_generator.samples}, Classes: {train_generator.class_indices}")
print(f"Validation dataset size: {val_generator.samples}, Classes: {val_generator.class_indices}")

# Model definition
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4, activation='softmax')  # 4 classes for rice diseases
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training loop
num_epochs = 10
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/rice_disease_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
)

# Print training and validation metrics
for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs - 1}, "
          f"Loss: {history.history['loss'][epoch]:.4f}, "
          f"Accuracy: {history.history['accuracy'][epoch]:.4f}, "
          f"Val Loss: {history.history['val_loss'][epoch]:.4f}, "
          f"Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

# Save the final model
model_path = "models/rice_disease_model.h5"
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"Model saved at {model_path}")