from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('beverage_can_model.h5')

# Define target image size
TARGET_SIZE = (300, 300)

# Function to preprocess the received image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route to receive images from Raspberry Pi
@app.route('/receive_image', methods=['POST'])
def receive_image():
    # Check if request contains image file
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'})

    # Get the image file from the request
    img_file = request.files['file']

    # Save the image file temporarily
    img_path = 'temp_image.jpg'
    img_file.save(img_path)

    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Make prediction
    prediction = model.predict(img_array)

    # Determine result based on prediction
    if prediction < 0.5:  # Example threshold for binary classification
        result = 'No error'
    else:
        result = 'Error'

    # Remove the temporary image file
    os.remove(img_path)

    # Send back the result to Raspberry Pi
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on port 5000
