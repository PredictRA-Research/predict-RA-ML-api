from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)
CORS(app)
# Load model architecture from JSON
with open('./model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load model architecture
loaded_model = model_from_json(loaded_model_json)

# Load model weights
loaded_model.load_weights("./model_weights.h5")

import random

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)

        # Make prediction
        prediction = loaded_model.predict(image)
        predicted_label = np.argmax(prediction)

        if predicted_label == 0:
            # Return a random number between 0 and 145
            random_number = random.randint(0, 145)
        else:
            # Return a random number between 145 and 224
            random_number = random.randint(145, 224)

        return jsonify({'prediction': random_number})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)


