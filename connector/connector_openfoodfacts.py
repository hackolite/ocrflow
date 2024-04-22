from retinanet.detect import  retina
from cropad.detect import crop_pad
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



def draw_rectangle():
	pass



def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str



def filter(boxes=None, shape=None):
	box_area_max = [0,0,0,0]
	box_aspectratio_max = [0,0,0,0]
	for box in boxes:
		if area(box) > area(box_area_max) and (box[3] > shape[0]/2):
			box_area_max = box
		
		if aspect_ratio(box) > aspect_ratio(box_aspectratio_max) and (box[1] > shape[0]/2):
			box_aspectratio_max = box

	return {"price":box_area_max, "ean":box_aspectratio_max }	




def walk_path(folder_path):
	#recognize text for price thanks to trOcr	
	for top, dirs, files in os.walk(folder_dataset):
		for file in files:
			if ".jpg" in file:
				file = os.path.join(top, file)
				im = cv2.imread(file)
				yield im	


def area(box):
	return (box[2] - box[0]) * (box[3] - box[1])



def aspect_ratio(box):
	try:
		res = (box[2] - box[0]) / (box[3] - box[1])
	except Exception as e :
		print(e)
		res = 0
	return res 



font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 1
color = (0, 0, 255) 


def draw_boxes(pad, boxes):
	for box in boxes:
			box = [int(i) for i in box]
			cv2.rectangle(pad,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)

def execute(crop_pad_detector=None, text_pad_detector=None, text_pad_recognizer=None, image_cv=None):
	#croppad
	box_max = [0,0,0,0]
	image_cv = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 3)

	resultat_yolo = crop_pad_detector.execute(image=image_cv, config=None)
	if len(resultat_yolo["boxes"]) > 0:
		boxes = resultat_yolo["boxes"]
		for box in boxes:
				box = [int(i) for i in box]
				cv2.rectangle(image_cv,(box[0],box[1]),(box[2],box[3]),(0,255,255),3)
				if area(box) > area(box_max):
					box_max = box

		pad = image_cv[box_max[1]:box_max[3], box_max[0]:box_max[2]]
	else:
		pad = image_cv 
	

	down_width = 500
	down_height = 400
	down_points = (down_width, down_height)
	pad = cv2.resize(pad, down_points, interpolation= cv2.INTER_LINEAR)
	text_pad = text_pad_detector.execute(image=pad, config=None)
	
	ean_code = "None"
	price = "None"
	if len(text_pad["boxes"]) > 0:
		boxes = text_pad["boxes"]
		boxes = filter(boxes, pad.shape)
		

		box = boxes["price"]
		box = [int(i) for i in box]
		resim  = pad[box[1]:box[3], box[0]:box[2]]
		price = text_pad_recognizer.execute_price(resim)

		box_ean = boxes["ean"]
		box_ean = [int(i) for i in box_ean]
		im_ean  = pad[box_ean[1]:box_ean[3], box_ean[0]:box_ean[2]]
		ean_code = text_pad_recognizer.execute_ean(im_ean)

	ean_code = ean_code.replace(" ","")
	ean_code = filtrer_caracteres_numeriques(ean_code)
	ean_file = "./{0}_pad.jpeg".format(get_random_string(10))
	cv2.imwrite(ean_file, pad)
	return {"ean":ean_code, "price":price, "path":ean_file}


def scan(folder=None):
	gen = walk_path(folder)
	for im in gen:
		resultat = execute(im)
		yield resultat 


def filtrer_caracteres_numeriques(chaine):
    caracteres_numeriques = ''.join(c for c in chaine if c.isdigit())
    return caracteres_numeriques
