import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2

# ✅ Load pretrained MobileNetV2 model and cut at the pooling layer
base_model = MobileNetV2(weights='imagenet', include_top=True)
model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[base_model.get_layer('global_average_pooling2d').output]  # 1280-dim features
)

def get_real_nn_steps(img_path):
    # ✅ Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Get feature vector from MobileNetV2
    features = model.predict(img_array)[0]  # shape: (1280,)
    
    # ✅ Create simulated weight matrix (for visualization only)
    weights = np.random.randn(10, 1280)  # 10 output neurons × 1280 features

    # ✅ Dot product
    dot_result = np.dot(weights, features).tolist()

    # ✅ Activation (ReLU-like)
    activated = [val if val > 0 else 0.01 * val for val in dot_result]

    # ✅ Softmax
    a = np.array(activated)
    exp_vals = np.exp(a - np.max(a))
    softmax_vals = (exp_vals / np.sum(exp_vals)).tolist()

    # ✅ Grayscale version for matrix view
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (28, 28))  # visual only
    matrix = gray_img.tolist()

    return {
        "matrix": matrix,
        "vector": features.tolist(),
        "weights": weights.tolist(),
        "dot": dot_result,
        "activated": activated,
        "softmax": softmax_vals
    }
