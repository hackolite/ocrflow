import matplotlib.pyplot as plt
import cv2
import keras_ocr
import PIL
from PIL import Image
import os
import time
import random
import string

#mode crop==False retourne l'image entière avec les coordonnées boxes relative a l'image
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)




class craft:
    def __init__(self, model_path=None):
        detector = keras_ocr.detection.Detector()
        if model_path != None:
            self.detector = detector.model.load_weights(model_path)
        else:
            self.detector = keras_ocr.detection.Detector(weights='clovaai_general')

    def model_constructor(self):
        pass


    def execute(self, image=None, config=None, crop=False, draw=False):

        boxes = self.detector.detect(images=[image])

        if crop == False:
            return {"image":image, "boxes":boxes}

