 
import matplotlib.pyplot as plt
import cv2
import keras_ocr
import PIL
from PIL import Image
import os
import time


#cv2.imread("./imgs/pad1643239971.2597027.jpeg")
detector = keras_ocr.detection.Detector(weights='clovaai_general')
image = keras_ocr.tools.read("./imgs/pad1643239971.2597027.jpeg")
boxes = detector.detect(images=[image])[0]

for box in boxes:
    try:
        xmin, ymin, xmax, ymax = int(box[0][0]) ,int(box[0][1]), int(box[2][0]), int(box[2][1])
    except Exception as e:
        print(e)
    cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),7)
cv2.imwrite("test_.jpg", image)
#boxes = detector.detect(images=[image], detection_threshold=0.7)[0]

