from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import random_split


# 1. Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 2. Load dataset
dataset = OxfordIIITPet(
    root="./data/raw",
    split="trainval",
    target_types="category",
    transform=transform,
    download=False
)


# 3. Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Number of classes:", len(dataset.classes))