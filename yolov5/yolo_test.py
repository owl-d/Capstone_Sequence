
from simple_detect import detect
from PIL import Image

import os
import time

import Pyro4

import threading

import base64
import numpy as np
import cv2 as cv
import json
yolo_ready_flag = True
json_class_path = './classify.json'
json_label_path = './label2name.json'
with open(json_class_path, 'r', encoding='utf-8') as f:
	json_data = json.load(f)
with open(json_label_path, 'r', encoding='utf-8') as f:
	json_label = json.load(f)

### Pyro4 Server Declaration ###
@Pyro4.expose
class server_callback(object):

	def response(self, data):

		global yolo_ready_flag

		if yolo_ready_flag == True:

			### Decode and Save RX image using OpenCV and numpy ############
			decoded_string = np.fromstring(base64.b64decode(data), np.uint8)

			decoded_img = cv.imdecode(decoded_string, cv.IMREAD_COLOR)

			cv.imwrite('from_android.jpg', decoded_img)

			print('Decoding Complete')
			###############################################################

		else:

			print('Ignoring HTTP RX')

### Create and Run Pyro4 server as an independent thread #########################
def create_pyro4Server():

	pyroDaemon = Pyro4.Daemon()

	ns = Pyro4.locateNS()

	uri = pyroDaemon.register(server_callback)

	ns.register('yolo', uri)

	print('Yolo Pyro4 Sever Ready')

	pyroDaemon.requestLoop()

pyro4ServerThread = threading.Thread(target=create_pyro4Server, args=())
pyro4ServerThread.daemon = True
pyro4ServerThread.start()
###############################################################################

while(True):

	if yolo_ready_flag == True:

		yolo_ready_flag = False
		box, max_name, label = detect(SOURCE = 'from_android.jpg',json_label=json_label,json_data=json_data)
		print('box : ', box)
		print('max_name :', max_name)
		print('label :', label)
		yolo_ready_flag = True
		time.sleep(7)
