from darkflow.net.build import TFNet
from collections import Counter
from multiprocessing import Pool
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import time
import os
import re
import argparse

# define constant
CONFIDENCE_RATE = 0.99
LICENSE_PLATE_COUNT = 10

# define the model options for YOLO and run
options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'threshold': 0.3,
    'load': 750
}
savedir = 'output'
input_dir = 'plate_image'

def save_img(img, savedir, name):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    save_path = os.path.join(savedir, name)
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    tfnet = TFNet(options)                                              # setup the model options

    for n, file in enumerate(os.scandir(input_dir)):
        print(file)
        img = cv2.imread(file.path)
        result = tfnet.return_predict(img)
        if len(result) > 0:
            h = result[0]['bottomright']['y'] - result[0]['topleft']['y']
            w = result[0]['bottomright']['x'] - result[0]['topleft']['x']
            crop_img = img[
                result[0]['topleft']['y']:result[0]['topleft']['y']+h, result[0]['topleft']['x']:result[0]['topleft']['x']+w]
            resized_crop = cv2.resize(crop_img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
            save_img(resized_crop, savedir, file.name)

            # add the box and label and display it
            tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
            br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
            img = cv2.rectangle(img, tl, br, (0,0,255), 7)
            text = result[0]['label'] + ', ' + str(result[0]['confidence'])
            img = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            resized = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
