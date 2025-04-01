import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ✅ Load pretrained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def predict_mobilenet(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # RGB resize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]

    return decoded  # [(id, label, prob), ...]
