import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))   


def extract_feature_vector(image_path):
	image = load_img(image_path, target_size=(224,224,3))
	image = img_to_array(image)
	image = preprocess_input(image)
	image = np.expands_dims(image, axis=0)
	feature_vector = model.predict(image)


folder_path = "./images"
image_path  = "image_path"
given_feature_vector = extract_feature_vector(image_path)

image_similarity_scores = {}

for filename in os.listdir(folder_path):
	if filename.endswith((".jpg", ".png", ".jpeg")):
		image_path = os.path.join(folder_path, filename)
		feature_vector = extract_feature_vector(image_path)
		similarity_score = cosine_similarity([given_feature_vector], [feature_vector])[0][0]
		image_similarity_scores[filename] =  similarity_score

top_10_scores = sorted(image_similarity_scores.items(), key=lambda x:x[1], reverse=True)[:10]


print(top_10_scores)


