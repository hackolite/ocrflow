import cv2
import os
import random
import string
from urllib.request import urlopen
from PIL import Image
import numpy as np
import configuration
from algorithms.cropad.detect import crop_pad
from algorithms.trocr.detect import trocr
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import logging
from annotation import boxe as BOXE
from configuration import models





#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

DEFAULT_WIDTH  = 500
DEFAULT_HEIGHT = 400

    
class ocr:


    def __init__(self, models=configuration.models):
        self.crop_pad_detector = crop_pad(model_path=models["tag_detect"]["yolo"])
        self.text_pad_detector = crop_pad(model_path=models["text_detect"]["yolo"])
        self.text_pad_detector.model.overrides = {'conf': 0.25, 'iou': 0.05, 'agnostic_nms': False, 'max_det': 1000}
        self.text_pad_recognizer = trocr(ean_model=models["recognition"]["trocr_ean"], price_model=models["recognition"]["trocr_price"])
        self.style = {"font": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 1, "thickness": 1, "color": (0, 0, 255)}
        self.annotation = BOXE.copy()


    def get_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def filter_boxes(self, boxes, shape):
        box_area_max = [0, 0, 0, 0]
        box_aspectratio_max = [0, 0, 0, 0]
        for box in boxes:
            if self.area(box) > self.area(box_area_max) and (box[3] > shape[0]/2):
                box_area_max = box
            
            if self.aspect_ratio(box) > self.aspect_ratio(box_aspectratio_max) and (box[1] > shape[0]/2):
                box_aspectratio_max = box
        return {"price": box_area_max, "ean": box_aspectratio_max}


    def area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def aspect_ratio(self, box):
        try:
            res = (box[2] - box[0]) / (box[3] - box[1])
        except ZeroDivisionError:
            res = 0
        return res 

    def execute(self, image_cv):
        self.annotation = BOXE.copy()
        box_max = [0, 0, 0, 0]
        image_cv = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 3)
        resultat_yolo = self.crop_pad_detector.execute(image=image_cv, config=None)
        

        if len(resultat_yolo["boxes"]) > 0:
            boxes = resultat_yolo["boxes"]
            for box in boxes:
                box = [int(i) for i in box]
                if self.area(box) > self.area(box_max):
                    box_max = box
            pad = image_cv[box_max[1]:box_max[3], box_max[0]:box_max[2]]
        else:
            pad = image_cv 

        self.annotation["shape_origin"]  = image_cv.shape
        self.annotation["position"]  = box_max
        self.annotation["boxes"]  = [self.annotation.copy(), self.annotation.copy()]
        self.annotation["boxes"][0]["shape_origin"] = pad.shape
        self.annotation["boxes"][1]["shape_origin"] = pad.shape


        resize_shape = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        HEIGHT, WIDTH, CHAN    = pad.shape
        pad = cv2.resize(pad, resize_shape, interpolation=cv2.INTER_LINEAR)
        text_pad = self.text_pad_detector.execute(image=pad, config=None)
        
        ean_code, price = "None", "None"
        if len(text_pad["boxes"]) > 0:
            boxes = text_pad["boxes"]
            boxes = self.filter_boxes(boxes, pad.shape)

            box_price = boxes["price"]
            box_price = [int(i) for i in box_price]
            box_price_ratio = [''] * 4
            box_price_ratio[0] = WIDTH/DEFAULT_WIDTH*box_price[0]
            box_price_ratio[2] = WIDTH/DEFAULT_WIDTH*box_price[2]            
            box_price_ratio[1] = HEIGHT/DEFAULT_HEIGHT*box_price[1]
            box_price_ratio[3] = HEIGHT/DEFAULT_HEIGHT*box_price[3]


            self.annotation["boxes"][1]["position"] = box_price_ratio
            box_price_pad= pad[box_price[1]:box_price[3], box_price[0]:box_price[2]]

            price = self.text_pad_recognizer.execute_price(box_price_pad)
            self.annotation["boxes"][1]["text"] = price            


            box_ean = boxes["ean"]
            box_ean = [int(i) for i in box_ean]
            box_ean_ratio = [''] * 4
            box_ean_ratio[0] = WIDTH/DEFAULT_WIDTH*box_ean[0]
            box_ean_ratio[2] = WIDTH/DEFAULT_WIDTH*box_ean[2]            
            box_ean_ratio[1] = HEIGHT/DEFAULT_HEIGHT*box_ean[1]
            box_ean_ratio[3] = HEIGHT/DEFAULT_HEIGHT*box_ean[3]


            self.annotation["boxes"][0]["position"] = box_ean_ratio

            box_ean_pad = pad[box_ean[1]:box_ean[3], box_ean[0]:box_ean[2]]
            ean_code = self.text_pad_recognizer.execute_ean(box_ean_pad)
            


        ean_code = ean_code.replace(" ", "")
        ean_code = self.filter_numeric_characters(ean_code)
        self.annotation["boxes"][0]["text"] = ean_code


        print({"ean": ean_code, "price": price, "annotation":self.annotation})
        return {"ean": ean_code, "price": price, "annotation":self.annotation}


    def extract_image(self, url_im):
        response = urlopen(url_im)
        image_data = response.read()
        image = Image.open(BytesIO(image_data))
        img = np.array(image)
        return img



    def process(self, url):
        im = self.extract_image(url)
        response = self.execute(im)
        return response


    def filter_numeric_characters(self, chaine):
        return ''.join(c for c in chaine if c.isdigit())
