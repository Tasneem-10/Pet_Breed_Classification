from locust import HttpUser, task, between


class PetBreedUser(HttpUser):

    wait_time = between(1, 2)

    def on_start(self):
        with open(
            "data/raw/oxford-iiit-pet/images/Abyssinian_1.jpg",
            "rb"
        ) as f:
            self.image_data = f.read()

    @task
    def predict(self):
        self.client.post(
            "/predict",
            files={
                "image": (
                    "Abyssinian_1.jpg",
                    self.image_data,
                    "image/jpeg"
                )
            }
        )