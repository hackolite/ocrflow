import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm



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

from torchvision.models.detection.anchor_utils import  DefaultBoxGenerator



def collate_fn(batch):
    return tuple(zip(*batch))


# Définissez la transformation pour le dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300))  # Convertir l'image en un tensor
])



dataset = TextDataset("../PAD/trainval.txt", image_dir="../PAD",transform=transform)
dataset.filter()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)


#anchor_generator = AnchorGenerator((16, 32, 64, 128, 256, 512), (0.1, 0.2, 0.4, 0.7, 1))





anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],steps=[8, 16, 32, 64, 100, 300],clip=True)
model = ssd300_vgg16(pretrained_backbone = True)  # Utilisez le modèle pré-entraîné ou False si vous voulez entraîner à partir de zéro
model.anchor_generator = anchor_generator

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
num_epochs = 500  # Définissez le nombre d'époques que vous souhaitez entraîner



for epoch in range(num_epochs):
    for images, targets in tqdm(data_loader):
        # Transférez les données sur le GPU si disponible
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Mettez le modèle en mode d'entraînement
        model.train()
        # Passez les données à travers le modèle
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) 
        # Effectuez la rétropropagation et mettez à jour les poids
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        loss_metric = losses.tolist()
        print(loss_metric)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses.tolist()}')

torch.save(model.state_dict(),f'ssd_{num_epochs}.pt')


