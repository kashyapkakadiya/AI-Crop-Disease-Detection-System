from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("crop_disease_model.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ==========================
# TREATMENT RECOMMENDATIONS
# ==========================
treatments = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericide spray and remove infected leaves.",
    "Pepper__bell___healthy": "The plant is healthy. No treatment required.",
    "Potato___Early_blight": "Apply fungicide and remove infected leaves.",
    "Potato___healthy": "The plant is healthy. No treatment required.",
    "Potato___Late_blight": "Use systemic fungicide and improve air circulation.",
    "Tomato_Bacterial_spot": "Use copper sprays and remove infected leaves.",
    "Tomato_Early_blight": "Apply fungicide and avoid overhead irrigation."
}

# ==========================
# IMAGE PREPARATION FUNCTION
# ==========================
def prepare_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ==========================
# ROUTES
# ==========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files['file']
    image = Image.open(file).convert("RGB")

    processed_image = prepare_image(image)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(float(np.max(prediction)) * 100, 2)

    treatment = treatments.get(predicted_class, "Consult agricultural expert.")

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        treatment=treatment
    )

# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    app.run(debug=True)