import streamlit as st 
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/807598/pexels-photo-807598.jpeg?cs=srgb&dl=pexels-sohail-nachiti-807598.jpg&fm=jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.title("Crop Disease Detection ")
st.subheader("Enter your Crop Image....")


def start(file):
    img_file_buffer = file
    image = Image.open(img_file_buffer)
    st.write("input image")
    # st.image(image)
    img_array = np.array(image) # if you want to pass it to OpenCV
    img = 'D:/Py/Bala_proj/color_img.jpg'
    cv2.imwrite(img, img_array)
    # st.image(image, caption="The caption", use_column_width=True)
    # array = np.reshape(img_array, (128, 128))

    if file:
        st.info("file entered")
        st.image(file)

    button = st.button('Enter')

    if button:
        model = load_model('D:/Py/Bala_proj/vgg16.h5')
        batch_size = 16
        image = cv2.imread(img)
        img = Image.fromarray(image)
        img = img.resize((128, 128))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        print(input_img)
        print(input_img.shape)
        i = input_img.reshape(-1,1)
        print("shape-i",i.shape)
            # result = model.predict_classes(input_img)
        result = model.predict(input_img)
        # st.write(result)
        st.subheader('The crop report is..')
        print(result)
        if result[0][0]:
            st.subheader("Disease State: Blight")
        elif result[0][1]:
            st.subheader("Disease state: Common Rust")
        elif result[0][2]:
            st.subheader("Disease state: Gray Leaf Spot")
        elif result[0][3]:
            st.subheader(" Plant is in Healthy state")

file = st.file_uploader("enter the image")


try:
    start(file)
except:
    pass
