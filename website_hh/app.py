import streamlit as st
from PIL import Image, ImageDraw,ImageFont
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
model = project.version(14).model



def load_images(cv_image, confidence_threshold, overlap_threshold):
    # Make prediction using Roboflow API
    robo_prediction = model.predict(cv_image, confidence=confidence_threshold*100, overlap=overlap_threshold*100).json()
    st.write(f'Our system detects {len(robo_prediction["predictions"])} humans in this photo')
    pil_img_with_boxes = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img_with_boxes)

    for prediction in robo_prediction['predictions']:
        label = prediction['class']
        xmin = prediction['x']
        ymin = prediction['y']
        width = prediction['width']
        height =  prediction['height']
        top_lx  = xmin  + prediction['width'] /2
        top_ly = ymin + prediction['height'] /2
        bottom_lx = xmin - prediction['width'] /2
        bottom_ly = ymin - prediction['height'] /2
        draw.rectangle(((top_lx, top_ly), (bottom_lx, bottom_ly)), outline='red', width=4)
        draw.text((bottom_lx, bottom_ly), label, fill='white')
    st.image(pil_img_with_boxes)


def main():
    st.header('Head Hunter')
    st.markdown('Counting crowds with confidence since 2023.')
    st.markdown("---")


    confidence_threshold = st.sidebar.slider('Confidence threshold:', 0.0, 1.0, 0.3, 0.01)
    overlap_threshold = st.sidebar.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)


    img_file_buffer = st.file_uploader('')


    if img_file_buffer is not None:
        img_bytes = img_file_buffer.getvalue()
        # Load uploaded image
        pil_image = Image.open(io.BytesIO(img_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        st.image(pil_image)
        if st.button("Predict Please...."):
            with st.spinner('Detecting Humans......'):
                time.sleep(10)
            load_images(cv_image, confidence_threshold, overlap_threshold)

        # pil_image = Image.open(io.BytesIO(r.content))
        # cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # st.image(cv_image, caption='Image with bounding boxes')


if __name__ == '__main__':
    main()
