from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # فعال کردن CORS

# Load model
model = tf.keras.models.load_model("eye_model.h5")
IMG_SIZE = (224, 224)
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_names[idx]

    return jsonify({"result": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
