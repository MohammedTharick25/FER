from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load the trained model
model = tf.keras.models.load_model('model_file_30epochs.h5')

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

@app.route('/')
def home():
    return render_template('index.html')

# Process uploaded images
@app.route('https://mohammedtharick25.github.io/FER/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = Image.open(file).convert('RGB')
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"emotion": "No Face Detected"})

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (48, 48)) / 255.0
        reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))

        result = model.predict(reshaped_face)
        label = np.argmax(result, axis=1)[0]

        return jsonify({"emotion": labels_dict[label]})

    return jsonify({"emotion": "Error processing image"})

# Process webcam images
@app.route('https://mohammedtharick25.github.io/FER/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    
    # Decode base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"emotion": "No Face Detected"})

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (48, 48)) / 255.0
        reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))

        result = model.predict(reshaped_face)
        label = np.argmax(result, axis=1)[0]

        return jsonify({"emotion": labels_dict[label]})

    return jsonify({"emotion": "Error processing image"})

if __name__ == '__main__':
    app.run(debug=True)
