import io
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

import app as app_module


class FakeModel:
    """Return deterministic logits without loading trained weights."""

    def __call__(self, image_tensor):
        logits = torch.zeros((image_tensor.shape[0], app_module.NUM_CLASSES))
        logits[:, 0] = 5.0
        return logits


def make_test_image():
    """Create a valid JPEG image in memory."""
    image = Image.new("RGB", (224, 224), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_health():
    with patch.object(app_module, "load_model", return_value=FakeModel()):
        with TestClient(app_module.app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_without_file():
    with patch.object(app_module, "load_model", return_value=FakeModel()):
        with TestClient(app_module.app) as client:
            response = client.post("/predict")

    assert response.status_code == 422


def test_predict_with_valid_image():
    with patch.object(app_module, "load_model", return_value=FakeModel()):
        with TestClient(app_module.app) as client:
            with make_test_image() as image_buffer:
                response = client.post(
                    "/predict",
                    files={
                        "file": (
                            "test.jpg",
                            image_buffer,
                            "image/jpeg",
                        )
                    },
                )

    assert response.status_code == 200

    result = response.json()
    assert result["breed"] == app_module.CLASS_NAMES[0]
    assert 0.0 <= result["confidence"] <= 1.0