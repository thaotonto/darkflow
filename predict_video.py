import cv2
from darkflow.net.build import TFNet
import numpy as np
import time


# define the model options and run

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'threshold': 0.3,
    'load': 750
}

tfnet = TFNet(options)
capture = cv2.VideoCapture('000002.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
max_confidence = 0
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            text = label + ', ' + str(result['confidence'])
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            if result['confidence'] > max_confidence:
              max_confidence = result['confidence']
            if result['confidence'] > 0.98:
              h = result['bottomright']['y'] - result['topleft']['y']
              w = result['bottomright']['x'] - result['topleft']['x']
              crop_img = frame[result['topleft']['y']:result['topleft']['y']+h, result['topleft']['x']:result['topleft']['x']+w]
              resized_crop = cv2.resize(crop_img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
              cv2.imwrite("frame%d.jpg" % time.time(), resized_crop)
        resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(max_confidence)
        capture.release()
        cv2.destroyAllWindows()
        break
