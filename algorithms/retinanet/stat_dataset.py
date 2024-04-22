import os
import pandas as pd
from torchvision.io import read_image
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from bs4 import BeautifulSoup
from PIL import Image
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import xmltodict


def getArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def parse_annotation(annotation_folder_path):
    '''
    Traverse the xml  pascalvoc
    '''
    boxes = []
    labels = []
    xml_path = annotation_folder_path
    with open(xml_path) as fd:
                doc = xmltodict.parse(fd.read())
                obj = doc["annotation"]["object"]
                orig_w = doc["annotation"]["size"]["width"]
                orig_h = doc["annotation"]["size"]["height"]
                boxes = []


                if type(obj) == dict:
                    xmin = int(obj["bndbox"]["xmin"])
                    ymin = int(obj["bndbox"]["ymin"])
                    xmax = int(obj["bndbox"]["xmax"])
                    ymax = int(obj["bndbox"]["ymax"])   
                    # get bboxes and their labels   
                    # rescale bboxes
                    bbox = [xmin, ymin, xmax, ymax]
                    if getArea(bbox) > 0:
                            boxes.append(bbox)
                            labels.append(0)

                else:
                    for i in obj:
                        xmin = int(i["bndbox"]["xmin"])
                        ymin = int(i["bndbox"]["ymin"])
                        xmax = int(i["bndbox"]["xmax"])
                        ymax = int(i["bndbox"]["ymax"])   
                        # get bboxes and their labels   
                        # rescale bboxes
                        bbox = [xmin, ymin, xmax, ymax]
                        if getArea(bbox) > 0:
                            boxes.append(bbox)
                            labels.append(0)

                return boxes, labels

def  parse(annotation_folder_path):
    boxes, labels = parse_annotation(annotation_folder_path)
    boxes = torch.as_tensor(boxes, dtype=torch.float32) 
    labels = torch.as_tensor(labels, dtype=torch.int64) 
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    return target




class TextDataset(Dataset):
    def __init__(self, annotation_path, image_dir=None, transform=None):
        self.data = pd.read_csv(annotation_path).values.tolist()
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    
    def filter(self):
        resultat = []
        for i in self.data:
            label_path = os.path.join(self.image_dir + "/Annotations", i[0] + ".xml")
            target = parse(label_path)
            if target["labels"].size(dim=0) > 0:
                resultat.append(i)
        print(len(resultat))        
        self.data = resultat


    def __getitem__(self, idx):

        file_image = self.data[idx][0] + ".jpg"
        file_label = self.data[idx][0] + ".xml"
        img_path = os.path.join(self.image_dir + "/Images", file_image)
        label_path = os.path.join(self.image_dir + "/Annotations", file_label)
        img = Image.open(img_path).convert("RGB")
        target = parse(label_path)
        to_tensor = torchvision.transforms.ToTensor()

        #if self.transform:
        #    img, transform_target = self.transform(np.array(img), np.array(target['boxes']))
        #    target['boxes'] = torch.as_tensor(transform_target)
        # change to tensor
        
        img = to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))



dataset = TextDataset("../PAD/trainval.txt", image_dir="../PAD")
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

for images, targets in data_loader:
    for box in targets[0]["boxes"].tolist():
        print((box[3]-box[1])/(box[2]-box[0]))