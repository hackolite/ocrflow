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


#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html

# Définir la fonction de chargement de données
def collate_fn(batch):
    return tuple(zip(*batch))


# Définissez la transformation pour le dataset
#transform = transforms.Compose([
#    transforms.GaussianBlur(kernel_size=501),
#    transforms.RandomRotation(degrees=180),
#    transforms.RandomHorizontalFlip(p=0.5),
#    transforms.Resize((300,300)),
#    transforms.ToTensor(),  # Convertir l'image en un tensor
#    # Ajoutez d'autres transformations si nécessaire
#])

# Définissez la transformation pour le dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300))  # Convertir l'image en un tensor
])


dataset = TextDataset("../PAD/trainval.txt", image_dir="../PAD",transform=transform)
dataset.filter()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

anchor_generator = AnchorGenerator((16, 32, 64, 128, 256, 512), (0.1, 0.2, 0.4, 0.7, 1))

retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=2, pretrained=False, pretrained_backbone = True, rpn_anchor_generator=anchor_generator)

retina.load_state_dict(torch.load("./retina_price_4.pt"))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_epochs = 10
retina.to(device)
    
# parameters
params = [p for p in retina.parameters() if p.requires_grad] # select parameters that require gradient calculation
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#optimizer = torch.optim.Adam(retina.parameters(), lr=0.01)


# about 4 min per epoch on Colab GPU
for epoch in range(num_epochs):
    retina.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = retina(images, targets) 
        losses = sum(loss for loss in loss_dict.values()) 
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(retina.parameters(), 0.1)
        optimizer.step()
        epoch_loss += losses
        loss_metric = losses.tolist()
        print(loss_metric)
    print(f'Epoch {epoch}/{num_epochs}, Loss: {loss_metric}')

torch.save(retina.state_dict(),f'retina_{num_epochs}.pt')