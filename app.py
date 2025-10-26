from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------- ایجاد اپلیکیشن ----------
app = Flask(__name__)
CORS(app)  # اجازه دسترسی از Frontend

# ---------- بارگذاری مدل ----------
model = tf.keras.models.load_model("eye_modelv2.h5")  # مسیر و نام مدل خودت
IMG_SIZE = (224, 224)
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ---------- تعریف endpoint /predict ----------
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    try:
        # پردازش تصویر
        img = Image.open(file).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # پیش‌بینی
        preds = model.predict(arr)[0]
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = class_names[idx]

        # خروجی JSON
        return jsonify({"result": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- اجرای سرویس ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
