import tensorflow as tf

MODEL_PATH = "banana_disease_resnet50_final.h5"
OUTPUT_PATH = "banana_disease_resnet50.tflite"

print("Loading H5 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("Conversion completed!")
print("Created:", OUTPUT_PATH)