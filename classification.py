import streamlit as st # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import io

# Load the trained model
model = load_model('/Users/shantanumislankar/Desktop/tomato/tomato_classifier_model.h5')

def prepare_image(img):
    img = img.resize((150, 150))  # Resize image to target size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

def classify_tomato(img):
    img = prepare_image(img)
    prediction = model.predict(img)
    if prediction[0] < 0.2:
        return "Premium"
    elif prediction[0] > 0.5:
        return "Rotten"
    else:
        return "Fresh"
    

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home", "About Project", "Prediction"])


if(app_mode == "Home"):
    st.header("TOMATO CLASSIFICATION SYSTEM")
    image_path = "tomato.jpg"
    st.image(image_path)

elif(app_mode == "About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text ("This dataset contains images of tomatoes:")
    st.code("fresh tomatoes")
    st.code("rotten tomatoes")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("train (4200 images each)")
    st.text("test (900 images each)")
    st.text("validation (900 images each)")

elif(app_mode == "Prediction"):
    st.header("Model Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        try:
            # Read the image file
            img = Image.open(uploaded_file)
            
            # Button to show image
            if st.button('Show Image'):
                st.image(img, caption='Uploaded Image.', use_column_width=True)
            
            # Button to classify image
            if st.button('Classify Image'):
                st.image(img, caption='Uploaded Image.', use_column_width=True)
                st.write("Classifying...")
                label = classify_tomato(img)
                st.write(f"The tomato is: {label}")
        
        except Exception as e:
            st.write(f"An error occurred: {e}")
