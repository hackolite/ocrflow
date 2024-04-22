import os
import cv2
import argparse
from glob import glob
import json
from xml.dom import minidom
import xml.etree.cElementTree as ET
import numpy as np
import shutil
import os
from sklearn.model_selection import train_test_split
import random
import xmltodict


parser = argparse.ArgumentParser(description='Converts the Pascal VOC 2012 dataset to a format suitable for training tbpp with this repo.')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--output', type=str, help='project_name')


args = parser.parse_args()
#assert os.path.exists(args.dataset_dir), "dataset_dir does not exist"


images_dir = os.path.join(args.dataset, "images")
labels_dir = os.path.join(args.dataset, "labels")
out_images_dir = os.path.join(args.output, "images")
out_labels_dir = os.path.join(args.output, "labels")
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)


def percent_list(data_list=None, percent_validation=None):
    dataset = data_list
    train = random.sample(dataset, int(len(dataset) * percent_validation))
    validation = list(set(dataset) - set(train))
    return train, validation

data = os.listdir(images_dir)
train ,validation = percent_list(data_list=data, percent_validation=0.8)


print(f"-- creating split files")
print(f"---- train.txt")
with open(os.path.join(args.output, "train.txt"), "w") as train_split_file:
        for sample in train:
            name = sample.replace(".jpg","").replace(".jpeg","")
            train_split_file.write(f"{sample} {name}.xml\n")

print(f"---- val.txt")
with open(os.path.join(args.output, "val.txt"), "w") as val_split_file:
        for sample in validation:
            name = sample.replace(".jpg","").replace(".jpeg","")
            val_split_file.write(f"{sample} {name}.xml\n")


print(f"---- trainval.txt")
with open(os.path.join(args.output, "split.txt"), "w") as trainval_split_file:
        train.extend(validation)
        for sample in train:
            #sample = sample.replace(".jpg","")
            name = sample.replace(".jpg","").replace(".jpeg","")
            trainval_split_file.write(f"{sample} {name}.xml\n")


print(f"-- copying images")
for i, sample in enumerate(list(glob(os.path.join(images_dir, "*jp*")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_images_dir, filename)
    )

print(f"-- copying labels")
for i, sample in enumerate(list(glob(os.path.join(labels_dir, "*xml")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_labels_dir, filename)
    )

#print(f"-- writing label_maps.txt")
with open(os.path.join(args.output, "label_maps.txt"), "w") as label_maps_file:
    labels = ["text"]
    for classname in labels:
        label_maps_file.write(f"{classname}\n")
