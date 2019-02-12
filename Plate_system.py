import cv2
from darkflow.net.build import TFNet
from collections import Counter
from multiprocessing import Pool
import numpy as np
import time
import os
import re
import argparse

import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
CONFIDENCE_RATE = 0.99
LICENSE_PLATE_COUNT = 10

showSteps = False
processing = False
licensePlate = []
DISPLAY_PLATE = False
plateToDisplay = None
WAIT_RESPONSE = False
###################################################################################################

# define the model options and run

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'threshold': 0.3,
    'load': 750
}

###################################################################################################
def process_image(imgOriginalScene):
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates
    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        # inform user no plates were found
        return False
    else:                                                       # else
        # if we get in here list of possible plates has at leat one plate

        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        licPlate = listOfPossiblePlates[0]
        licPlateNumber = ''
        allChars = ''
        for listOfPossiblePlate in listOfPossiblePlates:
            if len(listOfPossiblePlate.strChars) > 0:
                if re.match(r'\d{5}', listOfPossiblePlate.strChars):
                    licPlateNumber += listOfPossiblePlate.strChars
                elif re.match(r'(?<!\d)\d{4}(?!\d)$', listOfPossiblePlate.strChars):
                    licPlateNumber += listOfPossiblePlate.strChars
                elif re.match(r'\d{2}[A-Z]\d', listOfPossiblePlate.strChars):
                    licPlateNumber= ''.join((listOfPossiblePlate.strChars, licPlateNumber))


        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            return False                                          # and exit program
        # end if

    # end if else
    if re.match(r'\d{2}[A-Z]\d\d{5}', licPlateNumber) or re.match(r'\d{2}[A-Z]\d\d{4}', licPlateNumber):
        return licPlateNumber, imgOriginalScene
    else:
        return False
# end process image

def mycallback(result):
    global processing
    global DISPLAY_PLATE
    global plateToDisplay
    if result[0] != False:
        licPlateNumber, imgOriginalScene = result[0][0], result[0][1]
        if addPossiblePlate(licPlateNumber):
            DISPLAY_PLATE = True
            plateToDisplay = (licPlateNumber, imgOriginalScene)
            # cv2.imwrite(licPlateNumber + "-%d.jpg" % time.time(), imgOriginalScene)
    processing = False

def addPossiblePlate(licPlateNumber):
    global licensePlate
    licensePlate.append(licPlateNumber)
    c = Counter(licensePlate)
    if c.most_common(1)[0][1] >= LICENSE_PLATE_COUNT:
        licensePlate = []
        c.clear()
        return True
    else:
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                      # setup argument
    ap.add_argument("-vid", "--video", type=str)                          # argument for load video
    args = vars(ap.parse_args())

    tfnet = TFNet(options)                                          # setup the model options
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
    # end if
    pool=Pool()
    capture = cv2.VideoCapture(args["video"])                        # load video
    colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
    cv2.namedWindow('image')
    cv2.moveWindow("image", 20,20);
    max_confidence = 0
    camera = cv2.VideoCapture(0)
    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            results = tfnet.return_predict(frame)
            for color, result in zip(colors, results):
                if result['confidence'] > max_confidence:
                  max_confidence = result['confidence']
                if result['confidence'] > CONFIDENCE_RATE:
                    if processing == False and WAIT_RESPONSE == False:
                        processing = True
                        h = result['bottomright']['y'] - result['topleft']['y']
                        w = result['bottomright']['x'] - result['topleft']['x']
                        crop_img = frame[result['topleft']['y']:result['topleft']['y']+h, result['topleft']['x']:result['topleft']['x']+w]
                        resized_crop = cv2.resize(crop_img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
                        async_result = pool.map_async(process_image, (resized_crop,), callback=mycallback)

                # draw box on plate
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                frame = cv2.rectangle(frame, tl, br, color, 7)
                text = result['label'] + ', ' + str(result['confidence'])
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # show predict on screen
            resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
            if DISPLAY_PLATE == True:
                # send plate to server
                # WAIT_RESPONSE = True
                print('send licPlateNumber ' + plateToDisplay[0] + ' and plate image and camera image')
                cv2.imshow('image', plateToDisplay[1])
                ret_cam, frame_cam = camera.read()
                resized_cam = cv2.resize(frame_cam, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
                cv2.namedWindow("cam")
                cv2.moveWindow("cam", 20, 500)
                cv2.imshow("cam", resized_cam)
                plateToDisplay = None
                DISPLAY_PLATE = False
            else:
                cv2.imshow('frame', resized)
            # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(max_confidence)
            capture.release()
            camera.release()
            cv2.destroyAllWindows()
            pool.close()
            pool.join()
            break
