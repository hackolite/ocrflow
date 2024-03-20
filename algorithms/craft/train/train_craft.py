import zipfile
import datetime
import string
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
import pickle 
import imgaug
import keras_ocr

assert tf.test.is_gpu_available(), 'No GPU is available.'


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

c_font = []
c_background = []


for current_fonts, current_backgrounds in zip(font_splits, background_splits ):
         c_font.append(["./fonts/digitalnumbers/DigitalNumbers-Regular.ttf", "./fonts/sarpanch/Sarpanch-Medium.ttf"])
         c_background.append(current_backgrounds)


augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
    scale=(1.0, 1.2),
    rotate=(-5, 5)
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
])


image_generators = [
    keras_ocr.data_generation.get_image_generator(
        augmenter=augmenter,
        height=640,
        width=640,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        font_size=(60, 120),
        margin=50,
        rotationX=(-0.05, 0.05),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15)
    )  for current_fonts, current_backgrounds in zip(
        c_font,
        c_background
    )
]



def generate_dataset(folder=None, size=None, image_generators=None) :
  ind = 0
  dataset = []  
  for rs in image_generators[0]:
    dataset.append((rs[0], rs[1], 1))
    ind += 1
    if size == ind:
        break 
  return dataset


def generator_fusion():
    pass





if "dataset.pan" not in os.listdir("."):
    dataset = generate_dataset(folder="images", size=1000, image_generators=image_generators)
    with open("dataset.pan", "wb") as output_file:
        pickle.dump(dataset, output_file)
else:
    with open("dataset.pan", "rb") as output_file:
        dataset = pickle.load(output_file)


pickle_generator = iter(keras_ocr.data_generation.get_dataset_generator(dataset=dataset))



detector = keras_ocr.detection.Detector(weights='clovaai_general')
detector.model.load_weights("detector_2023-12-18T12:57:48.403733.h5")

detector_batch_size = 1
detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')

detection_train_generator  = detector.get_batch_generator(image_generator=pickle_generator, batch_size=detector_batch_size)


detector.model.fit_generator(
    generator=detection_train_generator,
    steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
    epochs=2,
    workers=0,
    callbacks=[
    tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=12),
    tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
    ]
)
