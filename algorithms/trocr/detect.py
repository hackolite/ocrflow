from transformers import VisionEncoderDecoderModel
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


class trocr:
	def __init__(self, size="large", ean_model='microsoft/trocr-large-printed', price_model="/home/lamaaz/xretail_plateform/models/trocr/content/models/price"):
			if torch.cuda.is_available() :
				self.device = "cuda" 
			else :
				self.device = "cpu"
			print(self.device)
			self.processor = TrOCRProcessor.from_pretrained(ean_model)
			self.model_price = VisionEncoderDecoderModel.from_pretrained(price_model)
			self.model_ean = VisionEncoderDecoderModel.from_pretrained(ean_model)
			self.model_ean.to(self.device)
			self.model_price.to(self.device)
			

	def model_constructor(self):
		pass


	def execute_price(self, image):
		pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
		generated_ids = self.model_price.generate(pixel_values)
		generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		return generated_text

	def execute_ean(self, image):
		pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
		generated_ids = self.model_ean.generate(pixel_values)
		generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		return generated_text