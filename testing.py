from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('/Users/shantanumislankar/Desktop/tomato/tomato_classifier_model.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load image with target size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

def classify_tomato(img_path):
    img = prepare_image(img_path)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print(f"The tomato is rotten.")
    else:
        print(f"The tomato is fresh.")

# Test the function
image_path = '/Users/shantanumislankar/Desktop/tomato/test/fresh/python_original_20230104_121811.jpg_f9b0d7f1-3285-494e-bb53-12cf9c70ac77.jpg'
classify_tomato(image_path)
