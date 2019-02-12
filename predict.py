import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

# define the model options and run

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'threshold': 0.3,
    'load': 750
}

tfnet = TFNet(options)
# read the color image and covert to RGB

img = cv2.imread('./sample_img/000003.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)
print(result);
img.shape
# pull out some info from the results

tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
h = result[0]['bottomright']['y'] - result[0]['topleft']['y']
w = result[0]['bottomright']['x'] - result[0]['topleft']['x']
label = result[0]['label']
confidence = result[0]['confidence']
text = '{}: {:.0f}%'.format(label, confidence * 100)
# add the box and label and display it
crop_img = img[result[0]['topleft']['y']:result[0]['topleft']['y']+h, result[0]['topleft']['x']:result[0]['topleft']['x']+w]
resized = cv2.resize(crop_img, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
cv2.imwrite("frame.jpg", resized)

cv2.imshow('predict', resized)
cv2.waitKey(0)
