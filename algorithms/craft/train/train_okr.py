import zipfile
import datetime
import string
import glob
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
import cv2
import keras_ocr

import os
import math
import imgaug
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import tensorflow as tf
import pickle 


assert tf.test.is_gpu_available()

data_dir = "."
detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')
alphabet = string.digits + string.ascii_letters + '!?. '
recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))


fonts = keras_ocr.data_generation.get_fonts(alphabet=alphabet,cache_dir=data_dir)
backgrounds = keras_ocr.data_generation.get_backgrounds(cache_dir=data_dir)
text_generator = keras_ocr.data_generation.get_text_generator(alphabet=alphabet)
print('The first generated text is:', next(text_generator))



def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test

background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

res = []
background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

c_font = []
c_background = []


for current_fonts, current_backgrounds in zip(font_splits, background_splits):
         #for i in current_fonts:
         #    if i == "./fonts/digitalnumbers/DigitalNumbers-Regular.ttf":
         #        print(i)
         #filtered_fonts = [font for font in current_fonts if  "./fonts/digitalnumbers/DigitalNumbers-Regular.ttf" == font]
         c_font.append(["./fonts/digitalnumbers/DigitalNumbers-Regular.ttf", "./fonts/sarpanch/Sarpanch-Medium.ttf"])
         c_background.append(current_backgrounds)
         c_font.append(current_fonts)


image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        font_size=(60, 120),
        margin=50
    )  for current_fonts, current_backgrounds in zip(
        font_splits,
        background_splits
    )
]

    
def generate_dataset(folder=None, size=None, imagen=None) :
  ind = 0
  dataset = []  
  for rs in image_generators[0]:
    dataset.append((rs[0], rs[1], 1))
    ind += 1
    if size == ind:
        break 
  return dataset


if "dataset.pan" not in os.listdir("."):
    dataset = generate_dataset(folder="images", size=1000, imagen=image_generators)
    with open("dataset.pan", "wb") as output_file:
        pickle.dump(dataset, output_file)
else:
    with open("dataset.pan", "rb") as output_file:
        dataset = pickle.load(output_file)

train, validation = sklearn.model_selection.train_test_split(
    dataset, train_size=0.8, random_state=42
)




