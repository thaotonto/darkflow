import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'threshold': 0.3,
    'load': 750
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

if int(major_ver)  < 3 :
    fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = capture.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

# Number of frames to capture
num_frames = 120;


print("Capturing {0} frames".format(num_frames))

# Start time
start = time.time()

# Grab a few frames
for i in range(0, num_frames) :
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
        cv2.imshow('frame', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# End time
end = time.time()

# Time elapsed
seconds = end - start
print("Time taken : {0} seconds".format(seconds))

# Calculate frames per second
fps  = num_frames / seconds;
print("Estimated frames per second : {0}".format(fps))
capture.release()

# while True:
#     stime = time.time()
#     ret, frame = capture.read()
#     if ret:
#         results = tfnet.return_predict(frame)
#         for color, result in zip(colors, results):
#             tl = (result['topleft']['x'], result['topleft']['y'])
#             br = (result['bottomright']['x'], result['bottomright']['y'])
#             label = result['label']
#             confidence = result['confidence']
#             text = '{}: {:.0f}%'.format(label, confidence * 100)
#             frame = cv2.rectangle(frame, tl, br, color, 5)
#             frame = cv2.putText(
#                 frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#         resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
#         cv2.imshow('frame', resized)
#         print('FPS {:.1f}'.format(1 / (time.time() - stime)))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()
