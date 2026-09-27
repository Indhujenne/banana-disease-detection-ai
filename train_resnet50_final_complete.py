import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# =============================
# PARAMETERS
# =============================

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 5

train_dir = "train"
val_dir = "val"
test_dir = "test"

# =============================
# DATA AUGMENTATION
# =============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# =============================
# LOAD DATASETS
# =============================

train_raw = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_raw.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_raw.map(lambda x,y:(data_augmentation(x),y))
train_dataset = train_dataset.prefetch(AUTOTUNE)

val_dataset = val_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# =============================
# BUILD MODEL
# =============================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# CALLBACKS
# =============================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "best_banana_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =============================
# TRAIN PHASE 1
# =============================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =============================
# FINE TUNING
# =============================

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =============================
# LOAD BEST MODEL
# =============================

model = tf.keras.models.load_model("best_banana_model.h5")

print("\nBest model loaded")

# =============================
# TEST ACCURACY
# =============================

test_loss, test_acc = model.evaluate(test_dataset)

print("\nTest Accuracy:", test_acc*100)

# =============================
# PER CLASS ACCURACY
# =============================

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nPer-class accuracy:")

for i,name in enumerate(class_names):

    mask = (y_true == i)

    acc = np.sum(y_pred[mask] == i) / np.sum(mask) * 100

    print(name,":",acc)

# =============================
# PLOT ACCURACY AND LOSS
# =============================

train_acc = history.history["accuracy"] + fine_history.history["accuracy"]
val_acc = history.history["val_accuracy"] + fine_history.history["val_accuracy"]

train_loss = history.history["loss"] + fine_history.history["loss"]
val_loss = history.history["val_loss"] + fine_history.history["val_loss"]

# Accuracy Graph

plt.figure(figsize=(8,6))

plt.plot(train_acc,label="Train Accuracy")
plt.plot(val_acc,label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_graph.png")

# Loss Graph

plt.figure(figsize=(8,6))

plt.plot(train_loss,label="Train Loss")
plt.plot(val_loss,label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_graph.png")

plt.show()

# =============================
# SAVE MODEL
# =============================

model.save("banana_disease_resnet50_final.h5")

print("\nModel saved successfully")