import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"


print("Loading training dataset...")

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("Loading validation dataset...")

val_dataset = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Loading test dataset...")

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_dataset.class_names

print("Detected classes:")
print(class_names)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        "Expected 6 classes, but found " + str(len(class_names))
    )


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


print("Building ResNet50 model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint = ModelCheckpoint(
    "best_banana_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=0.0000001,
    verbose=1
)


print("==========================================")
print("STARTING 6-CLASS TRAINING")
print("==========================================")
print("Classes:", class_names)
print("Epochs:", EPOCHS)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


print("Saving final model...")

model.save("banana_disease_resnet50_final.h5")


print("Loading best model...")

best_model = tf.keras.models.load_model(
    "best_banana_model.h5"
)


print("Testing model...")

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)


print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


print("Generating predictions...")

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


print("Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


print("==========================================")
print("TRAINING COMPLETED")
print("==========================================")
print("Test Accuracy:", test_accuracy)
print("Classes:", class_names)
print("Saved: best_banana_model.h5")
print("Saved: banana_disease_resnet50_final.h5")