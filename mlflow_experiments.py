import torch
import mlflow
import mlflow.pytorch

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet


# Configuration

BATCH_SIZE = 32
EPOCHS = 1
NUM_CLASSES = 37

LEARNING_RATES = [
    0.0001,
    0.0003,
    0.0005,
    0.001,
    0.003
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Dataset

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

print("Dataset size:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# MLflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Pet_Breed_Classification")


# Experiments

for lr in LEARNING_RATES:

    print("\n" + "=" * 50)
    print("Running experiment with learning rate:", lr)
    print("=" * 50)

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    with mlflow.start_run():

        # Parameters
        mlflow.log_params({
            "model": "ResNet50",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": lr,
            "num_classes": NUM_CLASSES,
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "split_seed": 42
        })

        # Training

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

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

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

        # Metrics
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Save model
        model_path = f"resnet50_lr_{lr}.pth"

        torch.save(
            model.state_dict(),
            model_path
        )

        mlflow.log_artifact(
            model_path,
            artifact_path="weights"
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print("Run completed.")