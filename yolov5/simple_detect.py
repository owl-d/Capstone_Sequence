import time
from flask import Flask, request, jsonify
import base64
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
from glob import glob
from pathlib import Path
import sys
import yaml
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync

json_class_path = './classify.json'
json_label_path = './label2name.json'
with open(json_class_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
with open(json_label_path, 'r', encoding='utf-8') as f:
    json_label = json.load(f)

yolo_ready_flag = True
WEIGHTS = 'best.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def detect():
    global yolo_ready_flag
    global WEIGHTS
    global IMG_SIZE
    global DEVICE
    global AUGMENT
    global CONF_THRES
    global IOU_THRES
    global CLASSES
    global AGNOSTIC_NMS

    if request.method == 'POST':
        box_ = []
        labels_ = []
        max_name = []
        _returns = {'start' : 'blank'}
        if yolo_ready_flag == True:

            yolo_ready_flag = False

            weights, imgsz = WEIGHTS, IMG_SIZE

            # Initialize
            device = select_device(DEVICE)
            half = device.type != 'cpu'  # half precision only supported on CUDA
            # print('device:', device)

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

            # Load image
            ### Decode base64 ################################
            image_str = request.data
            decoded_string = np.fromstring(base64.b64decode(image_str), np.uint8)
            img0 = cv2.imdecode(decoded_string, cv2.IMREAD_COLOR)
            img0 = cv2.resize(img0, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            # assert img0 is not None, 'Image Not Found ' + source

            # Padded resize
            img = letterbox(img0, imgsz, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t0 = time_sync()
            pred = model(img, augment=AUGMENT)[0]
            # print('pred shape:', pred.shape)

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

            # Process detections
            det = pred[0]
            # print('det shape:', det.shape)

            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            labels = []
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # if os.path.isfile(txt_path + '.txt'):
                #     os.remove(txt_path + '.txt')

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    box_.append(xywh)
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    label = names[int(cls)]
                    labels.append(label)

                # print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

            for idx in labels:
                labels_.append(json_label[str(idx)])
            _len_list = []
            for index in list(json_data.keys()):
                intersection = [index for idx in labels_ if len(list(set(json_data[str(index)]) & {idx}))]
                _len_list.append(len(intersection))
            max_name.append(list(json_data.keys())[_len_list.index(max(_len_list))])

            ### make dictionary for return value ###############################
            _returns = {'box' : box_, 'max_name' : max_name, 'labels' : labels_}
            ####################################################################

            yolo_ready_flag = True
            print(box_,max_name,labels_)

        return jsonify(_returns)

    else:
        return "GET or else"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
