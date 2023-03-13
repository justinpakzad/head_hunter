import streamlit as st
from PIL import Image, ImageDraw
import requests
from dotenv import load_dotenv
import os
import glob
import cv2
import random
from roboflow import Roboflow ###
import io
import base64
import time
import numpy as np



url = 'https://infer.roboflow.com/crowd_counting/12'

load_dotenv()





rf = Roboflow(api_key=st.secrets['api_key'])
project = rf.workspace().project('crowd_counting')
model = project.version(12).model


def load_images(cv_image, confidence_threshold, overlap_threshold):
    # Make prediction using Roboflow API
    robo_prediction = model.predict(cv_image, confidence=confidence_threshold*100, overlap=overlap_threshold*100).json()

    st.write(f'Our system detects {len(robo_prediction["predictions"])} humans')

    pil_img_with_boxes = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img_with_boxes)
    for prediction in robo_prediction['predictions']:
        label = prediction['class']
        xmin = prediction['x']
        ymin = prediction['y']
        xmax = xmin + prediction['width']
        ymax = ymin + prediction['height']
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='red', width=3)
        draw.text((xmin, ymin), label, fill='red')
    st.image(pil_img_with_boxes, caption='Image with bounding boxes')

def main():
    st.header('Head Hunter')
    st.markdown('Counting crowds with confidence since 2023.')
    st.markdown("---")


    folders = [os.path.basename(path) for path in glob.glob('/Users/justinpak/Desktop/roboflow_images/*') if os.path.isdir(path)]
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
        load_images(cv_image, confidence_threshold, overlap_threshold)


    else:
        image_paths = glob.glob(f'/Users/justinpak/Desktop/roboflow_images/{selected_folder}/*.*')
        random_path = random.choice(image_paths)

        cv_image = cv2.imread(random_path)
        cv_image = cv2.resize(cv_image, (640, 640))

        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        st.image(pil_image)

        with st.spinner('Detecting Humans......'):
            time.sleep(10)
        load_images(cv_image, confidence_threshold, overlap_threshold)


if __name__ == '__main__':
    main()
