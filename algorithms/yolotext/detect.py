from ultralytics import YOLO
import cv2

import random
import string


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

class crop_text:
	def __init__(self, model_path=None):
		if model_path != None:
			self.model = YOLO(model_path)
		else:
			raise ValueError


	def model_constructor(self):
		pass


	def execute(self,  image=None, config=None, crop=False):
		results = self.model(image)
		boxes = results[0].boxes.xyxy.tolist()
		if crop == False:
			pass
		return {"image":image, "boxes":boxes}
