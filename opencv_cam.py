#!/usr/bin/env python

import cv2
import time
import numpy as np
import json
import requests
from PIL import Image
from io import BytesIO

BASE_URL = "http://ae6e8b7d.ngrok.io"
URL_CHECK_IN = BASE_URL + "/api/records/check-in"
URL_CHECK_OUT = BASE_URL + "/api/records/check-out"
URL_IMAGE = BASE_URL + "/api/photos/raw/%s"

if __name__ == '__main__' :

    # Start default camera
    video = cv2.VideoCapture(0);
    image = cv2.imread("image.jpg")

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


    # Number of frames to capture
    num_frames = 120;


    print ("Capturing {0} frames".format(num_frames))

    # Start time
    start = time.time()

    # ret, frame = video.read()
    # resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
    # cv2.imwrite("platePhoto" + "-%d.jpg" % start, image)
    # cv2.imwrite("driverPhoto" + "-%d.jpg" % start, image)
    # multiple_files = [('platePhoto', ("platePhoto" + "-%d.jpg" % start, open("platePhoto" + "-%d.jpg" % start, 'rb'), 'image/jpg')),
    #                   ('driverPhoto', ("driverPhoto" + "-%d.jpg", open("driverPhoto" + "-%d.jpg" % start, 'rb'), 'image/jpg'))]
    data = { "plateNumber": "1235"}
    # r = requests.post(URL_CHECK_IN, files=multiple_files, data=data)
    r = requests.patch(URL_CHECK_OUT, data=data)
    print(r.status_code)
    if r.status_code == 200:
        parsed = json.loads(r.content)
        print(parsed)
        result = requests.get(BASE_URL + URL_IMAGE % parsed['platePhotoId'])
        img = Image.open(BytesIO(result.content))
        img.show()
    cv2.waitKey(0)
    # # Grab a few frames
    # for i in range(0, num_frames) :
    #     ret, frame = video.read()
    #     if ret:
    #         resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
    #         image_list = resized.tolist()
    #         image = np.array(image_list)
    #         image = image.astype(np.uint8)
    #         cv2.imshow('frame', resized)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print( "Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds;
    print ("Estimated frames per second : {0}".format(fps))

    # Release video
    video.release()
