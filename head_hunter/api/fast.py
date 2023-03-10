from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import numpy as np
import cv2
import io
import tensorflow as tf
from PIL import Image

app = FastAPI()

app.state.model = tf.keras.models.load_model('head_hunter/model/baseline_model.h5',  compile=False)


# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):

    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    cv2_img = Image.fromarray(cv2_img).convert('L')
    image_test = cv2_img.resize((256, 256), resample=Image.BICUBIC)

    image_test_array = np.expand_dims(image_test, axis=-1)

    image_test_array = np.expand_dims(image_test_array, axis=0)
    image_test_array = image_test_array / 255.0
    model = app.state.model
    model.compile(loss='huber_loss',optimizer='adam',metrics=['mae'])
    the_prediction = model.predict(image_test_array)


    return int(the_prediction)
