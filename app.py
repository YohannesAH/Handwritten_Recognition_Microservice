# Importing important libraries
from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import cv2 as cv
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import re

# Initializing Flask
app = Flask(__name__)

# Load the pre-trained model
model = load_model('mymodel')

# Extraction starts here
def split_and_extract_digits(image_path):
    try:
        # Read the image from the provided path
        image = Image.open(image_path)
        image = np.array(image)

        # Convert the image to grayscale
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur for noise reduction (adjust the kernel size as needed)
        blurred_image = cv.GaussianBlur(gray_image, (3, 3), 0)

        # Apply a fixed threshold to binarize the blurred image (adjust threshold as needed)
        threshold_value = 150  # Adjust the threshold value as needed
        _, binary_image = cv.threshold(blurred_image, threshold_value, 255, cv.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        digit_info = []

        for i, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv.boundingRect(contour)

            # Filter out small or non-digit-like regions based on area and aspect ratio
            area = cv.contourArea(contour)
            aspect_ratio = w / h

            # Adjust these parameters as needed based on your image
            min_area = 400  # Adjusted for better extraction of digit 6
            max_aspect_ratio = 1.0  # Adjusted to be less restrictive

            if min_area < area and aspect_ratio < max_aspect_ratio:
                # Extract the region containing a digit
                digit = binary_image[y:y + h, x:x + w]

                # Dilate the digit to make it bolder (adjust the kernel size as needed)
                kernel = np.ones((7, 7), np.uint8)
                dilated_digit = cv.dilate(digit, kernel, iterations=1)

                # Resize the digit to be slightly smaller (e.g., 20x20)
                digit = cv.resize(dilated_digit, (17, 17))

                # Calculate the position to paste the smaller digit in the center
                x_offset = 14 - digit.shape[1] // 2
                y_offset = 14 - digit.shape[0] // 2

                # Create the 28x28 canvas
                canvas = np.zeros((28, 28), dtype=np.uint8)

                # Paste the smaller digit in the center of the canvas
                canvas[y_offset:y_offset + digit.shape[0], x_offset:x_offset + digit.shape[1]] = digit

                # Convert the canvas to a Pillow Image
                digit_image = Image.fromarray(canvas)

                # Adjust the brightness of the extracted digit
                factor = 1.2  # Adjust the brightness factor as needed
                enhanced_digit = ImageEnhance.Brightness(digit_image).enhance(factor)

                digit_info.append({
                    'position': x,  # Use x-coordinate for position
                    'digit': np.array(enhanced_digit),
                })

        # Sort digit_info based on x-coordinate before returning
        digit_info.sort(key=lambda x: x['position'])

        return digit_info

    except Exception as e:
        return []

# Extraction ends here

# Prediction happens here
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64-encoded image string from JSON data
        try:
            data = request.get_json()
            base64_image = data['image_data']
        except KeyError:
            return jsonify({'error': 'Invalid JSON data'})

        if base64_image:
            try:
                # Extract the base64 data from the string
                image_data = re.sub('^data:image/.+;base64,', '', base64_image)
                image_data = base64.b64decode(image_data)
                image_path = 'temp_image.png'  # Specify a temporary image path

                with open(image_path, 'wb') as f:
                    f.write(image_data)

                # Extract individual digits from the image
                digit_info = split_and_extract_digits(image_path)

                predictions = []

                for digit_info in digit_info:
                    digit_image = (digit_info['digit'] / 255) - 0.5
                    digit_image_tensor = np.expand_dims(digit_image, axis=0)

                    # Make predictions
                    prediction = model.predict(digit_image_tensor)
                    digit_class = int(np.argmax(prediction))  # Convert to regular Python int

                    predictions.append({
                        'position': digit_info['position'],
                        'digit_class': digit_class,
                    })

                # Returning prediction
                recognized_digits = {'recognized_digits': predictions}
                return jsonify({'message': 'Recognized digits:', 'recognized_digits': recognized_digits})

            except Exception as e:
                return jsonify({'error': f'Error decoding base64 image: {e}'})

        return jsonify({'error': 'No base64 image provided'})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'})

if __name__ == '__main__':
    app.run(debug=True)
