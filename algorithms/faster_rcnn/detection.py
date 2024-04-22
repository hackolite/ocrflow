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
import cv2
from PIL import Image

#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html
anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),aspect_ratios=(0.025, 0.05, 1.0)) 
retina = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes = 2, pretrained=False, pretrained_backbone = True, rpn_anchor_generator=anchor_generator)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
retina.to(device)
retina.load_state_dict(torch.load("./faster_rcnn_model.pth"))



def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    print(preds)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    
    return preds

with torch.no_grad():
        im = cv2.imread("im.jpg")
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
            cv2.rectangle(img_data,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),3)
        cv2.imwrite("test.jpg", img_data)