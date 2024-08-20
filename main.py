from flask import Flask, request, jsonify
import base64
from flask import request, send_file
from flask import after_this_request
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
from functools import wraps
import time
import io
import requests
import gc
import random
import base64
from PIL import Image
import logging
import cv2
from source.cartoonize import Cartoonizer
import os
from cartoonifymod import Cartoonifier
import tensorflow as tf  
from flask_cors import CORS  
if not os.path.exists("gunicorn.err"):
    with open("gunicorn.err", "w") as f:
        f.write("")
    
if not os.path.exists("gunicorn.log"):
    with open("gunicorn.log", "w") as f:
        f.write("")
    


baseDir = os.path.abspath(os.path.dirname(__file__))


logging.basicConfig(level=logging.INFO,  # Set the desired logging level
                    # now format has acstime, level, location , message
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Log to the console
                        logging.FileHandler(os.path.join(baseDir, "cartoon.log"))  # Log to a file
                        # Add more handlers as needed, e.g., logging.FileHandler('app.log')
                    ])
def delete_file(file_path):
    try:
        os.remove(file_path)
        logging.info(f"File deleted: {file_path}")
    except Exception as e:
        logging.error(f"Error in delete_file: {e}")
        

app = Flask(__name__)
CORS(app)
# allow all origins
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}
baseDir = os.path.abspath(os.path.dirname(__file__))
global cartoonifier_model
cartoonifier_model = Cartoonifier()
logging.info("Models loaded successfully")
def before_first_request(f):
    already_run = False

    @wraps(f)
    def wrapper(*args, **kwargs):
        nonlocal already_run
        if not already_run:
            already_run = True
            f(*args, **kwargs)


    return wrapper
@app.before_request
@before_first_request
def load_model():
    try:
        # global cartoonify
        # cartoonify = Cartoonifier()
        logging.info("Models loaded successfully")
    except Exception as E:
        logging.error(f"Error in load_model : {E}")

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    try:
        data = request.json
        image = data['image']
        category = data['category'] # cartoon, artstyle, handdrawn, sketch
        # convert this base64string image into cv2 image
        image = Image.open(BytesIO(base64.b64decode(image)))
        logging.info(f'{image.size}')
        if category == "cartoon":
            image = cartoonifier_model.cartoonify(image)
        # elif category == "artstyle":
        #     image = cartoonifier_model.artstyleFunc(image)
        # elif category == "handdrawn":
        #     image = cartoonifier_model.handdrawnFunc(image)
        elif category == "sketch":
            image = cartoonifier_model.sketchFunc(image)
        elif category == "cartoon3d":
            image = cartoonifier_model.cartoon3dFunc(image)
        else:
            return jsonify({"message": "category not found"}), 400
        image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data = {
            "image": img_str

        }
        url = "http://35.193.78.211:5000"
        url = url + "/faceenhance"
        r = requests.post(url, json=data, timeout=120)
        image = r.json()['image']
        logging.info("Cartoonified successfully")
        gc.collect()
        return jsonify({"image": image}), 200
    except Exception as E:
        logging.error("failed to generate image: {E}")
        return jsonify({"message": "error something is wrong"}), 400

@app.route('/', methods=['GET'])
def initv2():
    try:
        gc.collect()
        # check if get request
        return "http://35.193.78.211:4545"
    except Exception as E:
        logging.error(f"Error in initv2 : {E}")
        return jsonify({"message": "Error: " + str(E)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4949)