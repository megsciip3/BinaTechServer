# app.py
from flask import Flask, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

# آدرس Space شما (دقت کن نام دقیق space را بگذاری)
HF_SPACE_API = "https://huggingface.co/spaces/megsciip/Bina_Tech/api/predict"

@app.route("/predict", methods=["POST"])
def predict():
    # دریافت فایل image از فرم-data
    if 'image' not in request.files:
        return jsonify({"error":"no image provided"}), 400

    file = request.files['image']
    # خواندن بایت‌های تصویر و تبدیل به base64
    b = file.read()
    b64 = base64.b64encode(b).decode('utf-8')

    # ساخت payload مطابق با API Space (data: [base64])
    payload = {"data": [b64]}

    try:
        # ارسال به Space (سرور به سرور)
        resp = requests.post(HF_SPACE_API, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": "error contacting HF Space", "detail": str(e)}), 500

    return jsonify(resp.json())
