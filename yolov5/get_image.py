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

### Libraries for Category Reader #######################
import os
import sys
from pathlib import Path
import json
import yaml
from glob import glob

### Categoray Reader Prep ################################
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


text_path = glob('./runs/detect/task' + '/*.txt')
json_path = './classify.json'
label2name_json_path = './label2name.json'
yaml_path = ROOT / 'file_path/data.yaml'


labels = []
data = yaml.load(open(yaml_path,'r',encoding = 'utf-8'),Loader=yaml.FullLoader)
labels.extend(data['names'])

index = list(range(1,len(labels)))
index2label_dict = dict(zip(index,labels))

json_data = json.load(open(label2name_json_path, 'r',encoding = 'utf-8'))

label_to_name = []
for idx in labels:
    label_to_name.append(json_data[idx])


### Category Reader Function Declaration #################
def Category(text_path = 'runs/detect/task/labels/',json_path = './classify.json'):
    with open(json_path, 'r',encoding = 'utf-8') as f:
        json_data = json.load(f)

    text_path = glob(text_path+'*.txt')
    print(text_path)
    max_name = []
    for path in text_path:
        with open(path, 'r') as tf:
            strings = tf.readlines()
        print(strings)
        pre_label = list(int(x.split(' ')[0]) for x in strings)
        labels = list(label_to_name[x] for x in pre_label)
        print(labels)
        _len_list = []
        for index in list(json_data.keys()):
            intersection = [index for idx in labels if len(list(set(json_data[str(index)]) & {idx}))]
            _len_list.append(len(intersection))
        max_name.append(list(json_data.keys())[_len_list.index(max(_len_list))])

    return max_name
##########################################################

inference_filename = '/home/ubuntu/capstone_project/yolov5/runs/detect/task/labels/from_android.txt'

inference_output_last_modified = int(os.path.getmtime(inference_filename))

publisher = Pyro4.Proxy('PYRONAME:yolo')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handel_request():

    global inference_filename
    global inference_output_last_modified

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

        ### Check whether inference output file has been changed ###
        inference_output_current_modified = int(os.path.getmtime(inference_filename))

        if inference_output_last_modified != inference_output_current_modified:
             bbox = Category()
             inference_output_last_modified = inference_output_current_modified
        ############################################################

        return bbox

    else :

        return "GET or else"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

