from flask import Flask, render_template, request, jsonify
from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)

MODEL_FILE = hf_hub_download(
    repo_id="jenneindhu/banana-disease-resnet50",
    filename="banana_disease_resnet50.tflite"
)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HISTORY_FILE = "predictions.json"

CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "moko_disease",
    "not_banana_leaf",
    "panama_disease",
    "yellow_sigatoka"
]

DISEASE_INFO = {
    "black_sigatoka": {
        "name": "Banana Black Sigatoka Disease",
        "advice": "Remove severely infected leaves and improve field sanitation. Apply recommended fungicides when necessary.",
        "fertilizer": "Use balanced fertilizers with adequate potassium and maintain proper plant nutrition."
    },
    "healthy": {
        "name": "Banana Healthy Leaf",
        "advice": "The leaf appears healthy. Continue regular monitoring and maintain proper irrigation and field sanitation.",
        "fertilizer": "Use balanced NPK fertilizer according to soil requirements."
    },
    "moko_disease": {
        "name": "Banana Moko Disease",
        "advice": "Remove and properly destroy infected plants. Avoid moving contaminated soil, tools, or planting materials between fields.",
        "fertilizer": "Maintain balanced plant nutrition and avoid excessive fertilizer application."
    },
    "panama_disease": {
        "name": "Banana Panama Disease",
        "advice": "Remove severely affected plants and maintain field sanitation.",
        "fertilizer": "Maintain balanced nutrition, especially adequate potassium, based on soil requirements."
    },
    "yellow_sigatoka": {
        "name": "Banana Yellow Sigatoka Disease",
        "advice": "Remove badly infected leaves and improve field sanitation. Apply recommended fungicide management when required.",
        "fertilizer": "Use balanced fertilizers and maintain adequate potassium for healthy leaf development."
    }
}


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def save_history(item):
    history = load_history()
    history.insert(0, item)
    history = history[:20]

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


print("Downloading TFLite model from Hugging Face...")

interpreter = Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully!")
print("Input shape:", input_details[0]["shape"])
print("Input type:", input_details[0]["dtype"])
print("Output shape:", output_details[0]["shape"])
print("Output type:", output_details[0]["dtype"])


@app.route("/")
def home():
    previous = load_history()
    return render_template("index.html", previous=previous)


@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        language = request.form.get("language", "English")

        filename = (
            datetime.now().strftime("%Y%m%d_%H%M%S_")
            + file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(image).astype(np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        interpreter.set_tensor(
            input_details[0]["index"],
            image_array
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        predicted_index = int(np.argmax(predictions))

        confidence = float(
            predictions[predicted_index]
        ) * 100

        predicted_class = CLASS_NAMES[predicted_index]

        if predicted_class == "not_banana_leaf":

            prediction_name = "Not a Banana Leaf"

            advice = (
                "The uploaded image does not appear to be "
                "a banana leaf. Please upload a clear "
                "banana leaf image."
            )

            fertilizer = (
                "No fertilizer recommendation is available "
                "because the image is not classified as a "
                "banana leaf."
            )

        else:

            info = DISEASE_INFO[predicted_class]

            prediction_name = info["name"]
            advice = info["advice"]
            fertilizer = info["fertilizer"]

        history_item = {
            "disease": prediction_name,
            "confidence": round(confidence, 2),
            "language": language,
            "advice": advice,
            "fertilizer": fertilizer,
            "image": "uploads/" + filename,
            "date": datetime.now().strftime("%d-%m-%Y %I:%M %p")
        }

        save_history(history_item)

        return jsonify({
            "success": True,
            "prediction": prediction_name,
            "class": predicted_class,
            "confidence": round(confidence, 2),
            "advice": advice,
            "fertilizer": fertilizer,
            "image": "uploads/" + filename
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )