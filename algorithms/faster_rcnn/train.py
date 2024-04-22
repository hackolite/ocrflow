import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import torchvision
from dataset import TextDataset

#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html

# Définir la fonction de chargement de données
def collate_fn(batch):
    return tuple(zip(*batch))



transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 



torch.cuda.empty_cache()
dataset = TextDataset("../PAD/trainval.txt", image_dir="../PAD", transform=transform)
dataset.filter()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

num_epochs = 5

anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),aspect_ratios=(0.025, 0.05, 1.0))    


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Définir le modèle Faster R-CNN avec une ancre génératrice
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Supposons que vous avez deux classes (background et objet d'intérêt)
num_classes = 2
# Obtenez le nombre d'entrées nécessaires pour la couche tête du classificateur
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Remplacez la tête du classificateur avec un nouveau classificateur
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Définissez les paramètres d'entraînement
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


if torch.cuda.is_available():
    model.cuda()

# Mettez le modèle en mode d'entraînement
model.train()

# Entraînez le modèle
num_epochs = 5
for epoch in range(num_epochs):
    for images, targets in data_loader:
        # Transférez les données sur l'appareil GPU si disponible
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Effacez les gradients précédents
        optimizer.zero_grad()

        # Passez les images au modèle
        loss_dict = model(images, targets)
        # Calculez la perte totale
        loss = sum(loss for loss in loss_dict.values())
        loss_metric  = loss.tolist()
        # Rétropropagation et mise à jour des paramètres
        loss.backward()
        optimizer.step()
        print(loss_dict)
    # Mettez à jour le scheduler d'apprentissage
    lr_scheduler.step()

    # Affichez la perte à chaque époque
    print(f'Epoch {epoch}/{num_epochs}, Loss: {loss_metric}')

# Sauvegardez le modèle entraîné
torch.save(model.state_dict(), 'faster_rcnn_model.pth')
