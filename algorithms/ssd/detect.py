import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


import cv2
import torch
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import transform as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD
import torchvision
from dataset import TextDataset
import time 
from functools import partial
import torchvision.transforms as transforms
from tqdm import tqdm 
from PIL import Image
from torchvision.models.detection.anchor_utils import  DefaultBoxGenerator

#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html



def model_constructor(config=None):
    pass



def make_prediction(model, img, threshold):
    threshold = 0.9
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds


def execute(model_path=None, image=None, config=None):
    with torch.no_grad():
        anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],steps=[8, 16, 32, 64, 100, 300],clip=True)
        retina = ssd300_vgg16(pretrained_backbone = True)  # Utilisez le modèle pré-entraîné ou False si vous voulez entraîner à partir de zéro
        retina.anchor_generator = anchor_generator
        #retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 2, pretrained=False, pretrained_backbone = True, rpn_anchor_generator=anchor_generator)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        retina.to(device)
        retina.load_state_dict(torch.load(model_path))
        im = image
        # You may need to convert the color.
        img_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_data)
        to_tensor = torchvision.transforms.ToTensor()
        im = to_tensor(im)
        im = im.to(device) 
        im = im.unsqueeze(0)
        preds = make_prediction(retina, im, 0.5)
        list_of_boxes = preds[0]["boxes"].tolist()
        for box in list_of_boxes:
            #print((box[3]-box[1])/(box[2]-box[0]))
            cv2.rectangle(img_data,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),3)
        #return {"image":img_data, "boxes":list_of_boxes}
        cv2.imwrite("image.jpeg",img_data)

im = cv2.imread("zeycpsweoi.jpg")
im = cv2.resize(im, (300,300))
execute(model_path="./ssd_500.pt", image=im, config=None)