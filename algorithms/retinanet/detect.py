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
from .dataset import TextDataset
import time
import cv2
from PIL import Image

#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html




class retina:
    def __init__(self, model_path=None):
        anchor_generator = AnchorGenerator((16, 32, 64, 128, 256, 512), (0.1, 0.2, 0.4, 0.7, 1))
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 2, pretrained=False, pretrained_backbone = True, rpn_anchor_generator=anchor_generator)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        if model_path != None:
            self.model.load_state_dict(torch.load(model_path))




    def model_constructor(self, config=None):
        pass



    def make_prediction(self, img, threshold):
        self.model.eval()
        preds = self.model(img)
        for id in range(len(preds)) :
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']) :
                if score > threshold : #select idx which meets the threshold
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list]
            preds[id]['labels'] = preds[id]['labels'][idx_list]
            preds[id]['scores'] = preds[id]['scores'][idx_list]

        return preds


    def execute(self, image=None, config=None):
        torch.cuda.empty_cache()
        with torch.no_grad():
            im = image
            # You may need to convert the color.
            img_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_data)
            to_tensor = torchvision.transforms.ToTensor()
            im = to_tensor(im)
            im = im.to(self.device) 
            im = im.unsqueeze(0)
            preds = self.make_prediction(im, 0.5)
            list_of_boxes = preds[0]["boxes"].tolist()
            return {"image":img_data, "boxes":list_of_boxes}