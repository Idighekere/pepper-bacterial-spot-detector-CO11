import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from flask import Flask, request, render_template

app = Flask(__name__)

MODEL_PATH = "models/custom_cnn_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['Bacterial_Spot', 'Healthy']


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    img_bytes = file.read()
    arr = preprocess(img_bytes)

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    label = CLASS_NAMES[pred_idx]
    confidence_pct = f"{confidence:.2%}"

    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    return render_template('index.html',
                           label=label,
                           confidence=confidence_pct,
                           image=img_b64)


if __name__ == '__main__':
    app.run(debug=True)