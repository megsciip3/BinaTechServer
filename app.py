from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ---------- ایجاد اپلیکیشن ----------
app = Flask(__name__)
CORS(app)

# ---------- بارگذاری مدل TFLite ----------
tflite_model_path = "eye_modelv2.tflite"  # فایل باید کنار app.py باشد
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(f"فایل مدل پیدا نشد: {tflite_model_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # پیش‌بینی با TFLite
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = class_names[idx]

        return jsonify({"result": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- اجرای سرویس ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render پورت را ارائه می‌دهد
    app.run(host="0.0.0.0", port=port)