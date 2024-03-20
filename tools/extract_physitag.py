from retinanet.detect import  retina
from cropad.detect import crop_pad
#from craft.detect import  craft
from trocr.detect import  trocr
import cv2
import os 
import random
import string
import pytesseract
from PIL import Image
import  pandas as pd
import urllib
from urllib.request import urlopen

from PIL import Image
from io import BytesIO
import numpy as np
import re




def aspect_ratio(box):
	try:
		res = (box[2] - box[0]) / (box[3] - box[1])
	except Exception as e :
		print(e)
		res = 0
	return res 


def area(box):
	return (box[2] - box[0]) * (box[3] - box[1])

def filter(boxes=None, shape=None):
	box_area_max = [0,0,0,0]
	box_aspectratio_max = [0,0,0,0]
	for box in boxes:
		if area(box) > area(box_area_max) and (box[3] > shape[0]/2):
			box_area_max = box
		
		if aspect_ratio(box) > aspect_ratio(box_aspectratio_max) and (box[1] > shape[0]/2):
			box_aspectratio_max = box

	return {"price":box_area_max, "ean":box_aspectratio_max }	


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def walk_path(folder_path):
	#recognize text for price thanks to trOcr	
	for top, dirs, files in os.walk(folder_path):
		for file in files:
			if ".jpeg" in file:
				file = os.path.join(top, file)
				im = cv2.imread(file)
				yield im	




crop_pad_detector = crop_pad(model_path="/home/lamaaz/xretail_plateform/models/yolo/digital_pad_stjp.pt")
walk  = walk_path("/home/lamaaz/xretail_plateform/ETUDE_CLIENT/PHYSICAL_PAD_DATASET/")


for im in walk:
	im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 3)
	resultat_yolo = crop_pad_detector.execute(image=im, config=None)
	if len(resultat_yolo["boxes"]) > 0:
		boxes = resultat_yolo["boxes"]
		print(boxes)
		box_max = boxes[0]
		box_max  = [int(i) for i in box_max]
		pad = im[box_max[1]:box_max[3], box_max[0]:box_max[2]]
		cv2.imwrite("{}.jpeg".format(get_random_string(10)), pad)

	