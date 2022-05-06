from flask import Flask, request
from flask import jsonify
from flask import redirect, url_for, send_from_directory, render_template
from time import sleep
import base64
import io
from PIL import Image

import Pyro4    # Interprocess communication library / Commnication between python processes

import cv2 as cv
import numpy as np



publisher = Pyro4.Proxy('PYRONAME:yolo')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handel_request():

    if request.method == 'POST':

	### Decode base64 ################################
        image_str = request.data
        decoded_string = np.fromstring(base64.b64decode(image_str), np.uint8)
        decoded_img = cv.imdecode(decoded_string, cv.IMREAD_COLOR)

        ### Encode image using OpenCV and numpy IO API in order to support Pyro4 TX

        retval, buffer = cv.imencode('.jpg', decoded_img)
        TX_data = base64.b64encode(buffer)

        ### Send base64 data through Pyro4 IPC pipeline ###
        publisher.response(TX_data.decode('utf-8'))
        ###################################################

        category = "milk"
        bbox = "2 150 150 180 180"

        return bbox

    else :

        return "GET or else"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

