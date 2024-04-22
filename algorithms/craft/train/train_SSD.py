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
import xmltodict



assert tf.test.is_gpu_available()

script_dir = "."

detector_basepath = os.path.join(script_dir, f'detector_{datetime.datetime.now().isoformat()}')


def generate_ln(file_type="xml", filepath=None):

    lines = []
    with open(filepath) as fd:
        doc = xmltodict.parse(fd.read())
        print(doc["annotation"]["object"])	
    
    
    if type(doc["annotation"]["object"]) == dict:
        coordinate = doc["annotation"]["object"]['bndbox']
        lines.append((np.array([[coordinate["xmin"], coordinate["ymin"]],  [coordinate["xmax"], coordinate["ymin"]],  [coordinate["xmax"], coordinate["ymax"]],  [coordinate["xmin"], coordinate["ymax"]]]).astype("float32"), "text"))

    elif type(doc["annotation"]["object"]) == list:
        for box in doc["annotation"]["object"]:
            coordinate = box['bndbox']
            lines.append((np.array([[coordinate["xmin"], coordinate["ymin"]],  [coordinate["xmax"], coordinate["ymin"]],  [coordinate["xmax"], coordinate["ymax"]],  [coordinate["xmin"], coordinate["ymax"]]]).astype("float32"), "text"))
    return lines
	
	
def generate_dataset(folder=None) :
  ind = 0
  dataset = []
  for fl in os.listdir(folder):
    try:	
        if "jpg" in fl:
            ln = "./"+folder+"/"+fl
            fp = ln.replace(".jpg",".xml")        
            lines = generate_ln(file_type="xml", filepath=fp)
            dataset.append((ln, [lines], 1))
            ind += 1
 
    except Exception as e:
        print(e)

  return dataset

dataset = generate_dataset(folder="pad")





def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test


train, validation = sklearn.model_selection.train_test_split(
    dataset, train_size=0.8, random_state=42
)

print(train)

augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
    scale=(1.0, 1.2),
    rotate=(-5, 5)
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
])

generator_kwargs = {'width': 640, 'height': 640}
training_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=train,
    augmenter=augmenter,
    **generator_kwargs
)


validation_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)


detector = keras_ocr.detection.Detector()

batch_size = 1
detector_batch_size = 1 
training_generator, validation_generator = [
    detector.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_generator, validation_image_generator]
]

detector.model.fit_generator(
    generator=training_generator,
    steps_per_epoch=math.ceil(len(train) / batch_size),
    epochs=1000,
    workers=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5), tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5'), tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv')],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation) / batch_size)
)
