import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_predict_without_file():
    with TestClient(app) as client:
        response = client.post("/predict")

        assert response.status_code == 422

def test_predict_with_valid_image():
    image_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "oxford-iiit-pet"
        / "images"
        / "Abyssinian_1.jpg"
    )

    with TestClient(app) as client:
        with image_path.open("rb") as image_file:
            response = client.post(
                "/predict",
                files={
                    "file": (
                        image_path.name,
                        image_file,
                        "image/jpeg",
                    )
                },
            )

    assert response.status_code == 200

    result = response.json()
    assert result["breed"] in __import__("app").CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0