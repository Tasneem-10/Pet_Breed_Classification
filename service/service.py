import bentoml
import torch

from PIL import Image
from torch import nn
from torchvision import models, transforms

from class_names import CLASS_NAMES


NUM_CLASSES = 37

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        "resnet18_pet_breed_optimized.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 60}
)
class PetBreedClassifier:

    @bentoml.api
    def predict(self, image: Image.Image) -> dict:

        image = image.convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        breed = CLASS_NAMES[predicted.item()]

        return {
            "breed": breed,
            "confidence": float(confidence.item())
        }