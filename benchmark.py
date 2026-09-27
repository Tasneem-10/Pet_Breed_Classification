import time
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

NUM_CLASSES = 37

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(
    "data/raw/oxford-iiit-pet/images/Abyssinian_1.jpg"
).convert("RGB")

image = transform(image)
image = image.unsqueeze(0).to(device)


def benchmark_model(model, model_name):

    model = model.to(device)
    model.eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(image)

    # Actual measurement
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(50):
            model(image)

    end = time.perf_counter()

    total_time = end - start
    average_time = total_time / 50

    print(f"\n{model_name}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average inference time: {average_time * 1000:.2f} ms")


# ResNet50
resnet50 = models.resnet50(weights=None)
resnet50.fc = nn.Linear(
    resnet50.fc.in_features,
    NUM_CLASSES
)

resnet50.load_state_dict(
    torch.load(
        "resnet50_pet_breed.pth",
        map_location=device
    )
)

benchmark_model(resnet50, "ResNet50")


# ResNet18
resnet18 = models.resnet18(weights=None)
resnet18.fc = nn.Linear(
    resnet18.fc.in_features,
    NUM_CLASSES
)

resnet18.load_state_dict(
    torch.load(
        "resnet18_pet_breed_optimized.pth",
        map_location=device
    )
)

benchmark_model(resnet18, "ResNet18")