# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

"""

import argparse
import os
import sys
from pathlib import Path

from glob import glob
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import check_requirements
from utils.torch_utils import time_sync
from Detect_new import run

check_requirements(exclude=('tensorboard', 'thop'))
filename = [ROOT / 'Detect_Source/*.jpg',ROOT / 'Detect_Source/*.jpg',ROOT / 'Detect_Source/*.jpg']
t1 = time_sync()
for i in filename:
    run(source=i)
    print('------------------------------------------------------------------------------------------------------------')
t2 = time_sync()-t1
print(t2)