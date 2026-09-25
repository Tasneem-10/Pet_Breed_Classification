import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image


NUM_CLASSES = 37

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load class names

dataset = OxfordIIITPet(
    root="./data/raw",
    split="trainval",
    target_types="category",
    download=False
)

class_names = dataset.classes


# Build model

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# Load trained weights

model.load_state_dict(
    torch.load(
        "resnet50_pet_breed.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# Image preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Prediction function

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    breed = class_names[predicted.item()]

    return breed, confidence.item()


# Test prediction

image_path = input("Enter image path: ")

breed, confidence = predict(image_path)

print("Predicted breed:", breed)
print(f"Confidence: {confidence * 100:.2f}%")