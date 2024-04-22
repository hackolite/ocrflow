
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
from pascal_voc_writer import Writer


crop_pad_detector = crop_pad(model_path="/home/lamaaz/xretail_plateform/models/yolo/digital_pad_stjp.pt")
text_pad_detector = crop_pad(model_path="/home/lamaaz/xretail_plateform/models/yolo/text_pad_stjp.pt")


def area(box):
	return (box[2] - box[0]) * (box[3] - box[1])



def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def execute(image_cv=None):
	#croppad
	box_max = [0,0,0,0]
	resultat_yolo = crop_pad_detector.execute(image=image_cv, config=None)
	if len(resultat_yolo["boxes"]) > 0:
		boxes = resultat_yolo["boxes"]
		for box in boxes:
				box = [int(i) for i in box]
				if area(box) > area(box_max):
					box_max = box

		pad = image_cv[box[1]:box[3], box[0]:box[2]]
	else:
		pad = image_cv 
	
	
	text_pad = text_pad_detector.execute(image=pad, config=None)
	boxes = text_pad["boxes"]

	pref = get_random_string(7)
	name = "./{0}_raw.jpeg".format(pref)
	cv2.imwrite(name, pad)
	im = cv2.imread(name)
	h, w, _ = im.shape
	writer = Writer(name, w, h)
	
	for box in boxes:
		# ::addObject(name, xmin, ymin, xmax, ymax)
		writer.addObject('text', box[0], box[1], box[2], box[3])
	writer.save("./{0}_raw.xml".format(pref))



def walk_path(folder_path):
	#recognize text for price thanks to trOcr	
	for top, dirs, files in os.walk(folder_path):
		for file in files:
			if ".jpeg" in file:
				file = os.path.join(top, file)
				im = cv2.imread(file)
				execute(im)	


if __name__ == "__main__":
	walk_path("./ETUDE_ST_JOSEPH_RAW")