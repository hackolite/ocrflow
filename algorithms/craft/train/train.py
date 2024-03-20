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
import numpy as np

import keras_ocr

assert tf.test.is_gpu_available()

alphabet = string.digits + string.ascii_letters + '!?.'
recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
fonts = keras_ocr.data_generation.get_fonts(cache_dir=".")
backgrounds = keras_ocr.data_generation.get_backgrounds(cache_dir=".")
text_generator = keras_ocr.data_generation.get_text_generator(alphabet=alphabet)


print('The first generated text is:', next(text_generator))


def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test


background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

background_splits = background_splits
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

# See what the first validation image looks like.
image, lines = next(image_generators[1])
text = keras_ocr.data_generation.convert_lines_to_paragraph(lines)
print('The first generated validation image (below) contains:', text)

detector = keras_ocr.detection.Detector(weights='clovaai_general')


detector_batch_size = 1
detector_basepath = os.path.join(".", f'detector_{datetime.datetime.now().isoformat()}')
detection_train_generator, detection_val_generator, detection_test_generator = [
    detector.get_batch_generator(
        image_generator=image_generator,
        batch_size=detector_batch_size
    ) for image_generator in image_generators
]


detector.model.fit(
    detection_train_generator, 
    steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
    ]
)