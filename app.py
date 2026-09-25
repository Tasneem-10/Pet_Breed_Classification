import torch
from torch import nn
from torchvision import models, transforms
from class_names import CLASS_NAMES
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io


NUM_CLASSES = 37

app = FastAPI(
    title="Pet Breed Classification API"
)


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




# Build model

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# Load trained model

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


# Health endpoint

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

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