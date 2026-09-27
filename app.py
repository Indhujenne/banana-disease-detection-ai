from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
from openpyxl import Workbook, load_workbook

app = Flask(__name__)

UPLOAD_FOLDER = "static"
EXCEL_FILE = "results.xlsx"
MODEL_FILE = "banana_disease_resnet50_final.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# LOAD RESNET50 MODEL
# --------------------------------------------------

print("Loading ResNet50 model...")
model = tf.keras.models.load_model(MODEL_FILE)
print("Model loaded successfully!")

# Exact class order from your train folders
CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "moko_disease",
    "panama_disease",
    "yellow_sigatoka"
]

# Display names
DISPLAY_NAMES = {
    "black_sigatoka": "Black Sigatoka Disease",
    "healthy": "Healthy Leaf",
    "moko_disease": "Moko Disease",
    "panama_disease": "Panama Disease",
    "yellow_sigatoka": "Yellow Sigatoka Disease"
}

# --------------------------------------------------
# ADVICE
# --------------------------------------------------

ADVICES = {
    "black_sigatoka": {
        "English": [
            "Remove severely infected leaves and maintain good field sanitation.",
            "Improve air circulation by maintaining proper plant spacing.",
            "Use recommended fungicides according to agricultural guidelines."
        ],
        "Telugu": [
            "తీవ్రంగా సోకిన ఆకులను తొలగించి పొలాన్ని శుభ్రంగా ఉంచండి.",
            "మొక్కల మధ్య సరైన దూరం ఉంచి గాలి ప్రసరణను మెరుగుపరచండి.",
            "వ్యవసాయ నిపుణుల సూచనల ప్రకారం శిలీంద్ర నాశక మందులను ఉపయోగించండి."
        ],
        "Hindi": [
            "गंभीर रूप से संक्रमित पत्तियों को हटाएं और खेत को साफ रखें.",
            "पौधों के बीच उचित दूरी रखकर हवा का संचार बेहतर करें.",
            "कृषि विशेषज्ञ की सलाह के अनुसार फफूंदनाशक का उपयोग करें."
        ],
        "Tamil": [
            "கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றி வயலை சுத்தமாக வைத்திருக்கவும்.",
            "செடிகளுக்கு இடையில் சரியான இடைவெளி வைத்து காற்றோட்டத்தை மேம்படுத்தவும்.",
            "விவசாய நிபுணர் ஆலோசனைப்படி பூஞ்சைநாசினிகளை பயன்படுத்தவும்."
        ]
    },

    "moko_disease": {
        "English": [
            "Remove and destroy infected plants to prevent disease spread.",
            "Avoid moving contaminated soil or plant material between fields.",
            "Maintain good field sanitation."
        ],
        "Telugu": [
            "వ్యాధి సోకిన మొక్కలను తొలగించి నాశనం చేయండి.",
            "కలుషితమైన మట్టి లేదా మొక్కల భాగాలను ఇతర పొలాలకు తరలించవద్దు.",
            "పొలాన్ని పరిశుభ్రంగా ఉంచండి."
        ],
        "Hindi": [
            "संक्रमित पौधों को हटाकर नष्ट करें.",
            "दूषित मिट्टी या पौधों के हिस्सों को दूसरे खेतों में न ले जाएं.",
            "खेत की स्वच्छता बनाए रखें."
        ],
        "Tamil": [
            "பாதிக்கப்பட்ட செடிகளை அகற்றி அழிக்கவும்.",
            "மாசுபட்ட மண் அல்லது தாவரப் பகுதிகளை வேறு வயல்களுக்கு கொண்டு செல்ல வேண்டாம்.",
            "வயலை சுத்தமாக வைத்திருக்கவும்."
        ]
    },

    "panama_disease": {
        "English": [
            "Remove severely infected plants and avoid spreading contaminated soil.",
            "Use healthy planting material from reliable sources.",
            "Maintain proper drainage and field sanitation."
        ],
        "Telugu": [
            "తీవ్రంగా సోకిన మొక్కలను తొలగించండి మరియు కలుషితమైన మట్టి వ్యాప్తిని నివారించండి.",
            "నమ్మకమైన వనరుల నుండి ఆరోగ్యకరమైన నాట్లను ఉపయోగించండి.",
            "సరైన నీటి పారుదల మరియు పొల పరిశుభ్రతను పాటించండి."
        ],
        "Hindi": [
            "गंभीर रूप से संक्रमित पौधों को हटाएं और दूषित मिट्टी के प्रसार को रोकें.",
            "विश्वसनीय स्रोतों से स्वस्थ रोपण सामग्री का उपयोग करें.",
            "उचित जल निकासी और खेत की स्वच्छता बनाए रखें."
        ],
        "Tamil": [
            "கடுமையாக பாதிக்கப்பட்ட செடிகளை அகற்றி மாசுபட்ட மண் பரவுவதைத் தடுக்கவும்.",
            "நம்பகமான மூலங்களிலிருந்து ஆரோக்கியமான நடவு பொருட்களை பயன்படுத்தவும்.",
            "சரியான வடிகால் மற்றும் வயல் சுகாதாரத்தை பராமரிக்கவும்."
        ]
    },

    "yellow_sigatoka": {
        "English": [
            "Remove severely affected leaves and maintain good field sanitation.",
            "Improve air circulation around plants.",
            "Use recommended fungicides according to agricultural guidelines."
        ],
        "Telugu": [
            "తీవ్రంగా ప్రభావితమైన ఆకులను తొలగించి పొలాన్ని శుభ్రంగా ఉంచండి.",
            "మొక్కల చుట్టూ గాలి ప్రసరణను మెరుగుపరచండి.",
            "వ్యవసాయ నిపుణుల సూచనల ప్రకారం శిలీంద్ర నాశక మందులను ఉపయోగించండి."
        ],
        "Hindi": [
            "गंभीर रूप से प्रभावित पत्तियों को हटाएं और खेत को साफ रखें.",
            "पौधों के आसपास हवा का संचार बेहतर करें.",
            "कृषि विशेषज्ञ की सलाह के अनुसार फफूंदनाशक का उपयोग करें."
        ],
        "Tamil": [
            "கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றி வயலை சுத்தமாக வைத்திருக்கவும்.",
            "செடிகளைச் சுற்றி காற்றோட்டத்தை மேம்படுத்தவும்.",
            "விவசாய நிபுணர் ஆலோசனைப்படி பூஞ்சைநாசினிகளை பயன்படுத்தவும்."
        ]
    }
}

# --------------------------------------------------
# FERTILIZERS
# --------------------------------------------------

FERTILIZERS = {
    "black_sigatoka": {
        "Potassium": "Recommended dose based on soil test",
        "Magnesium": "Use according to soil requirement"
    },

    "moko_disease": {
        "Balanced NPK": "Use according to soil test",
        "Organic manure": "Apply well-decomposed manure"
    },

    "panama_disease": {
        "Potassium": "Use according to soil requirement",
        "Organic manure": "Apply well-decomposed manure"
    },

    "yellow_sigatoka": {
        "Potassium": "Recommended dose based on soil test",
        "Magnesium": "Use according to soil requirement"
    }
}

# --------------------------------------------------
# CREATE EXCEL FILE
# --------------------------------------------------

if not os.path.exists(EXCEL_FILE):

    wb = Workbook()
    ws = wb.active

    ws.append([
        "Image",
        "Disease",
        "Confidence",
        "Language",
        "Advice",
        "Fertilizer"
    ])

    wb.save(EXCEL_FILE)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():

    wb = load_workbook(EXCEL_FILE)
    ws = wb.active

    previous = []

    for row in ws.iter_rows(min_row=2, values_only=True):

        previous.append({
            "image": os.path.join(
                app.config["UPLOAD_FOLDER"],
                row[0]
            ),

            "disease": row[1],
            "confidence": row[2],
            "language": row[3],
            "advice": row[4],
            "fertilizer": row[5]
        })

    return render_template(
        "index.html",
        previous=previous
    )


# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    language = request.form.get("language", "English")

    files = request.files.getlist("images")

    results = []

    wb = load_workbook(EXCEL_FILE)
    ws = wb.active

    for file in files:

        if file.filename == "":
            continue

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        # Load image
        img = Image.open(filepath).convert("RGB")

        # Resize exactly like training
        img = img.resize((224, 224))

        # Convert to numpy array
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index] * 100
        )

        class_name = CLASS_NAMES[predicted_index]

        disease_name = DISPLAY_NAMES[class_name]

        # Healthy leaf
        if class_name == "healthy":

            advice_list = {
                "English": [
                    "The banana leaf appears healthy. Continue regular monitoring and proper irrigation."
                ],
                "Telugu": [
                    "అరటి ఆకు ఆరోగ్యంగా కనిపిస్తోంది. సాధారణ పర్యవేక్షణ మరియు సరైన నీటి పారుదల కొనసాగించండి."
                ],
                "Hindi": [
                    "केले का पत्ता स्वस्थ दिखाई देता है. नियमित निगरानी और उचित सिंचाई जारी रखें."
                ],
                "Tamil": [
                    "வாழை இலை ஆரோக்கியமாக உள்ளது. வழக்கமான கண்காணிப்பு மற்றும் சரியான நீர்ப்பாசனத்தை தொடரவும்."
                ]
            }

            advice = advice_list.get(
                language,
                advice_list["English"]
            )

            fertilizers = [
                "No special fertilizer recommendation"
            ]

        else:

            advice = ADVICES[class_name].get(
                language,
                ADVICES[class_name]["English"]
            )

            fertilizers = [
                f"{key} ({value})"
                for key, value in FERTILIZERS[class_name].items()
            ]

        # Save result to Excel
        ws.append([
            filename,
            disease_name,
            round(confidence, 2),
            language,
            ", ".join(advice),
            ", ".join(fertilizers)
        ])

        results.append({

            "image": filepath,

            "disease": disease_name,

            "confidence": round(
                confidence,
                2
            ),

            "advice": advice,

            "fertilizer": fertilizers
        })

    wb.save(EXCEL_FILE)

    return {
        "results": results
    }


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )