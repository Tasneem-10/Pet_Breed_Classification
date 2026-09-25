import torch
from torchvision import models


NUM_CLASSES = 37


model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


print(model)