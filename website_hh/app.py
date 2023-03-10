import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os
import glob
import cv2
import random
from roboflow import Roboflow ###
import io
import base64
# ︻デ═一


# Background color
st.markdown('''
    <style>
    .css-ffhzg2 {
        position: absolute;
        background: rgb(0, 0, 0);
        background-color: rgb(188, 184, 244);
        color: rgb(250, 250, 250);
        inset: 0px;
        overflow: hidden;
        }
    </style>


            ''',
            unsafe_allow_html=True)

# Header color
st.markdown('''
    <style>
    .css-10trblm {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 600;
        color: rgb(41, 40, 54);
        letter-spacing: -0.005em;
        padding: 1rem 0px;
        margin: 0px;
        line-height: 1.2;
        }
    </style>
            ''',
            unsafe_allow_html=True)

# "Our prediction is" color
st.markdown('''
    <style>
    .css-13sdm1b {
        color: rgb(41, 40, 54);
        font-family: "Source Sans Pro", sans-serif;
        }
    </style>
            ''',
            unsafe_allow_html=True)

# "Upload an image" color
st.markdown('''
    <style>
    .css-1yjuwjr {
        color: rgb(41, 40, 54);
        font-family: "Source Sans Pro", sans-serif;
        }
    </style>
            ''',
            unsafe_allow_html=True)

# "Here's your image" color
st.markdown('''
    <style>
    .css-ltfnpr {
        color: rgb(41, 40, 54);
        font-family: "Source Sans Pro", sans-serif;
        }
    </style>
            ''',
            unsafe_allow_html=True)

# File name color and font size
st.markdown('''
    <style>
    .css-1uixxvy {
        color: rgb(41, 40, 54);
        font-family: "Source Sans Pro", sans-serif;
        font-size: 12px;
        }
    </style>
            ''',
            unsafe_allow_html=True)

# File size font size
st.markdown('''
    <style>
    .css-7oyrr6 {
        font-family: "Source Sans Pro", sans-serif;
        font-size: 12px;
        }
    </style>
            ''',
            unsafe_allow_html=True)

load_dotenv()

#url = 'https://headhunter-d6euadhcta-ew.a.run.app/upload_image'
url = 'http://localhost:8000/upload_image'

st.header('Head Hunter 😈')

st.markdown('''
            Counting crowds with confidence since 2023.
            ''')

st.markdown("---")


rf = Roboflow(api_key=st.secrets['api_key'])

project = rf.workspace().project('crowd_counting')
model = project.version(12).model

folder_names = []
folders = []

for path in glob.glob('/Users/User/code/denesito/justin/roboflow_images/*'):
    if os.path.isdir(path):
        folder_name = os.path.basename(path)
        folders.append(folder_name)


confidence_threshold = st.slider('Confidence threshold:', 0.0, 1.0, 0.5, 0.01)

overlap_threshold = st.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)
# probably different value

selected_folder = st.selectbox('What size crowd do you wanna see?', folders)




def load_images(crowd_size):

    image_paths = glob.glob(f'/Users/User/code/denesito/justin/roboflow_images/{crowd_size}/*.*')

    #random_paths = random.sample(image_paths, k=1)

    session_state = st.session_state
    if not session_state.get('random_paths'):
        session_state.random_paths = random.sample(image_paths, k=1)

    for path in session_state.random_paths:
        # Load the image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image)

        # Run the prediction on the image
        robo_prediction = model.predict(path, confidence=confidence_threshold*100, overlap=overlap_threshold*100).json()

        # Draw the bounding boxes on the image
        for prediction in robo_prediction['predictions']:
            x, y, w, h = map(int, [prediction['x'], prediction['y'], prediction['width'], prediction['height']])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert the image from BGR to RGB and display it
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Image with detection boxes")

        st.write(len(robo_prediction['predictions']))

load_images(selected_folder)


img_file_buffer = st.file_uploader('')

if img_file_buffer is not None:

    img_bytes = img_file_buffer.getvalue()

    #breakpoint()
    response = requests.post(url, files={'img': img_bytes})

    result = response.json()


if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption='')

    st.write(f"Our count is {result}. 😈")


#temp
