import streamlit as st
from image_classification import predict
from PIL import Image, ImageOps

st.title("Image Classification Dogs vs Cats")
st.header("Classification Example")
st.text("Upload a cat or dog Image for image classification")

uploaded_file = st.file_uploader("Choose a photo of cat or dog ...", type="jpg")
if uploaded_file is not None:
        image_path = Image.open(uploaded_file)
        st.image(image_path, caption='Uploaded img.', use_column_width=True)
        st.write("")
        st.write("Wait we are Classifying...")
        predict(image_path)

