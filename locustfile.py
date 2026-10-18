from pathlib import Path

from locust import HttpUser, between, task


IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "raw"
    / "oxford-iiit-pet"
    / "images"
    / "Abyssinian_1.jpg"
)


class PetBreedUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        if not IMAGE_PATH.is_file():
            raise FileNotFoundError(
                f"Test image not found: {IMAGE_PATH}"
            )

        with IMAGE_PATH.open("rb") as image_file:
            self.image_data = image_file.read()

    @task
    def predict(self):
        self.client.post(
            "/predict",
            files={
                "file": (
                    IMAGE_PATH.name,
                    self.image_data,
                    "image/jpeg",
                )
            },
        )
