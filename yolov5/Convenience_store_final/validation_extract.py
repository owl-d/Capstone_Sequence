# _*_ coding: utf_8 _*_

# ************ import module ************

import os
import cv2
import yaml
import json
import pandas as pd
import albumentations as A
from glob import glob
from tqdm import tqdm
import os
import shutil
import xml.etree.ElementTree as ET
from numba import jit
import cv2
import numpy as np
import random
import wandb
import torch
import random

from Yolov5_preprocessing import voc2yolo,yolo2voc,coco2yolo,yolo2coco,voc2coco,coco2voc,bbox_iou,clip_bbox,str2annot,annot2str,load_image
from File_Tool import zip_file_extract,copy_img_label
# from Data_Augmentation import Augmentation_

os.makedirs("/home/parkjuntae/바탕화면/CustomData/Valid", exist_ok=True)
result_dir = "/home/parkjuntae/바탕화면/CustomData/Valid/"

# Combine the images and labels in one folder
main_dir = "/home/parkjuntae/바탕화면/CustomData"
os.makedirs(main_dir + "/Valid_all",exist_ok= True)
Valid_all_path = main_dir + '/Valid_all'
pre_img_list = glob(result_dir + '*/*.jpg',recursive = True)
pre_label_list = glob(result_dir + '*/*.xml',recursive = True)

# ************ Image Label Combine ************
# print("img file copy")
copy_img_label(pre_img_list,pre_label_list,Valid_all_path)