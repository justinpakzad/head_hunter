import streamlit as st
from PIL import Image, ImageDraw
import requests
import os
import glob
import cv2
import random
from roboflow import Roboflow ###
import io
import base64
import time
import numpy as np
import matplotlib as plt
# ︻デ═一

url = 'https://detect.roboflow.com/crowd_counting/12'


rf = Roboflow(api_key=st.secrets['api_key'])
project = rf.workspace().project('crowd_counting')
model = project.version(12).model


def load_images(cv_image, confidence_threshold, overlap_threshold):

    buffer = io.BytesIO()
    Image.fromarray(cv_image).save(buffer, format='JPEG', quality=90)
    buffer.seek(0)

    # Base64 encode the image
    image_str = base64.b64encode(buffer.getvalue()).decode('ascii')

    # Make prediction using Roboflow API
    robo_prediction = model.predict(cv_image, confidence=confidence_threshold, overlap=overlap_threshold).json()

    st.write(f'Our system detects {len(robo_prediction["predictions"])} humans')

    for prediction in robo_prediction['predictions']:
        label = prediction['class']
        x = prediction['x']
        y = prediction['y']
        # xmax = xmin + prediction['width']
        # ymax = ymin + prediction['height']
        # draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='red', width=3)
        # draw.text((xmin, ymin), label, fill='red')
        draw.text((x,y),'red')
    st.image(pil_img_with_boxes, caption='Image with bounding boxes')

    ## Construct the URL to retrieve image.
    upload_url = f'https://detect.roboflow.com/crowd_counting/12?api_key={st.secrets["api_key"]}&confidence={confidence_threshold}&{overlap_threshold}=&format=image&labels=off'

    ## POST to the API.
    r = requests.post(upload_url,
                    data=image_str,
                    headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })

    return r


    folders = [os.path.basename(path) for path in glob.glob('/Users/justinpak/Desktop/roboflow_images/*') if os.path.isdir(path)]
    confidence_threshold = st.sidebar.slider('Confidence threshold:', 0.0, 1.0, 0.3, 0.01)
    overlap_threshold = st.sidebar.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)
    selected_folder = st.sidebar.selectbox('What size crowd do you wanna see?', folders)

def main():
    st.header('Head Hunter')
    st.markdown('Counting crowds with confidence since 2023.')
    st.markdown("---")


    folders = [os.path.basename(path) for path in glob.glob('/Users/User/code/denesito/justin/roboflow_images/*') if os.path.isdir(path)]
    confidence_threshold = st.sidebar.slider('Confidence threshold:', 0.0, 1.0, 0.3, 0.01)
    overlap_threshold = st.sidebar.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)
    selected_folder = st.sidebar.selectbox('What size crowd do you wanna see?', folders)


    img_file_buffer = st.file_uploader('')
    if img_file_buffer is not None:
        img_bytes = img_file_buffer.getvalue()

        # Load uploaded image
        pil_image = Image.open(io.BytesIO(img_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # st.image(pil_image)


        with st.spinner('Detecting Humans......'):
            time.sleep(10)
        r = load_images(cv_image, confidence_threshold, overlap_threshold)

        pil_image = Image.open(io.BytesIO(r.content))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        st.image(cv_image, caption='Image with bounding boxes')

    else:
        image_paths = glob.glob(f'/Users/User/code/denesito/justin/roboflow_images/{selected_folder}/*.*')
        random_path = random.choice(image_paths)

        cv_image = cv2.imread(random_path)


        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        st.image(pil_image)

        with st.spinner('Detecting Humans......'):
            time.sleep(10)
        r = load_images(cv_image, confidence_threshold, overlap_threshold)

        pil_image = Image.open(io.BytesIO(r.content))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        st.image(cv_image, caption='Image with bounding boxes')

if __name__ == '__main__':
    main()
