import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0001
NUM_CLASSES = 37

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = OxfordIIITPet(
    root="./data/raw",
    split="trainval",
    target_types="category",
    transform=transform,
    download=False
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# Optimized model: ResNet18
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


# Validation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


val_accuracy = 100 * correct / total

print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Save optimized model
torch.save(
    model.state_dict(),
    "resnet18_pet_breed_optimized.pth"
)

print("Optimized model saved successfully.")