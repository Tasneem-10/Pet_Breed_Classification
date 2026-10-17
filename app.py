import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from class_names import CLASS_NAMES


NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = Path(__file__).resolve().parent / "resnet18_pet_breed_optimized.pth"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = None


def load_model():
    """Load the trained model and its saved weights."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    loaded_model = models.resnet18(weights=None)
    loaded_model.fc = nn.Linear(
        loaded_model.fc.in_features,
        NUM_CLASSES,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    loaded_model.load_state_dict(state_dict)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    global model

    logger.info("Loading model from %s", MODEL_PATH)
    model = load_model()
    logger.info("Model loaded successfully on %s", device)

    yield

    model = None


app = FastAPI(
    title="Pet Breed Classification API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "num_classes": NUM_CLASSES,
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Uploaded file must be an image",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.verify()

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image",
        ) from None

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "breed": CLASS_NAMES[predicted.item()],
        "confidence": float(confidence.item()),
    }