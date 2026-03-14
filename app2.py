import streamlit as st
import joblib 
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model=load_model("animal_classifier.h5")
IMG_SIZE=128

def predict_image(image):
    img=np.array(image)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img=img/255.0
    img=np.reshape(img,(1,IMG_SIZE,IMG_SIZE,3))
    prediction=model.predict(img)

    if prediction[0][0]>0.5:
        return "DOG"
    else:
        return "CAT"

st.title("Animal Image Classifier")

st.write("Upload an image of a cat or a dog")

uploaded_img=st.file_uploader("Choose and image...",type=['jpg','jpeg','png'])

if uploaded_img is not None:
    image=Image.open(uploaded_img)
    st.image(image,caption="Uploaded Image")
    st.write("Predicting...")
    result=predict_image(image)
    st.success(f"Prediction: {result}")