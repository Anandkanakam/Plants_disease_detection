import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

model = load_model('final_model.keras')  # add final model file
class_dict = np.load("class_names.npy")  # labels file we have saved as array

def predict(image):
    IMG_SIZE = (1, 256, 256, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    accuracy = pred_proba[0][pred] * 100  # Get the accuracy percentage
    return pred, accuracy

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        base64_img = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_img}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

contnt = "<p>Herbal medicines are preferred in both developing and developed countries as an alternative to " \
         "synthetic drugs mainly because of no side effects. Recognition of these plants by human sight will be " \
         "tedious, time-consuming, and inaccurate.</p> " \
         "<p>Applications of image processing and computer vision " \
         "techniques for the identification of the medicinal plants are very crucial as many of them are under " \
         "extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial " \
         "for the conservation of biodiversity.</p>"

if __name__ == '__main__':
    add_bg_from_local("Background.jpg")
    new_title = '<p style="font-family:sans-serif; color:red; font-size: 50px;">Medicinal Leaf Classification</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    contnt = '<p style="font-family:sans-serif; color:white; font-size: 20px;">Herbal medicines are preferred in both developing and developed countries as an alternative to synthetic drugs mainly because of no side effects \
    Recognition of these plants by human sight will be tedious, time-consuming, and inaccurate.</p>'
    st.markdown(contnt,unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((300, 300))
        st.image(img)
        if st.button("Predict"):
            pred, accuracy = predict(img)
            name = class_dict[pred]
            result = f'<p style="font-family:sans-serif; color:Red; font-size: 32px;">The given image is {name} with an accuracy of {accuracy:.2f}%</p>'
            st.markdown(result, unsafe_allow_html=True)




