from flask import Flask, render_template, request, redirect, session
import numpy as np
import os
import json
import tensorflow as tf
import time
import cv2
from mobilenet_logic import get_real_nn_steps
from mobilenet_model import predict_mobilenet

app = Flask(__name__)

# ✅ Load the trained model
model = tf.keras.models.load_model('real_model.h5')

# ✅ Load class labels
with open('class_labels.json', 'r') as f:
    CLASS_LABELS = json.load(f)

app.secret_key = 'supersecretkey'  # ✅ required for using sessions

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    if not image_file:
        return "No file uploaded"

    # ✅ Save with timestamp to avoid caching issues
    filename = f"{int(time.time())}_{image_file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(path)

    session['latest_image'] = path

    # ✅ Show grayscale 28x28 matrix on index.html
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    matrix = img.tolist()

    return render_template('index.html', matrix=matrix)


@app.route('/vector', methods=['POST'])
def vector():
    image_path = session.get('latest_image')
    if not image_path or not os.path.exists(image_path):
        return "No image found"

    # ✅ Extract MobileNet steps
    steps = get_real_nn_steps(image_path)

    session['vector'] = steps['vector']
    session['weights'] = steps['weights']
    session['dot'] = steps['dot']
    session['activated'] = steps['activated']
    session['softmax'] = steps['softmax']

    return render_template('vector.html', matrix=steps['vector'])


@app.route('/weights', methods=['POST'])
def weights():
    vector_json = request.form.get('vector')
    if not vector_json:
        return "Vector not found", 400
    vector = json.loads(vector_json)

    weights = np.random.randint(-3, 4, (10, len(vector)))
    result = np.dot(weights, np.array(vector)).tolist()

    session['weighted_result'] = result

    return render_template('weights.html', vector=vector, weights=weights.tolist(), result=result)


@app.route('/activation')
def activation():
    result = session.get('weighted_result')
    if result is None:
        return "Dot product result not found"

    activated = [val if val > 0 else 0.01 * val for val in result]
    session['activated_result'] = activated

    return render_template('activation.html', result=result, activated=activated)


@app.route('/softmax')
def softmax():
    activated = session.get('activated_result')
    if activated is None:
        return "Activated result not found"

    a = np.array(activated)
    exp_vals = np.exp(a - np.max(a))
    softmax_vals = (exp_vals / np.sum(exp_vals)).tolist()
    session['softmax_result'] = softmax_vals

    return render_template('softmax.html', softmax=softmax_vals)


@app.route('/predict')
def predict():
    image_path = session.get('latest_image')
    if not image_path or not os.path.exists(image_path):
        return "No image uploaded"

    results = predict_mobilenet(image_path)
    top_label = results[0][1]
    top_confidence = round(results[0][2] * 100, 2)

    prediction_text = f"{top_label} ({top_confidence}%)"
    return render_template('prediction.html', prediction=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
    