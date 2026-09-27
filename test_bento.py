import requests

image_path = "data/raw/oxford-iiit-pet/images/Abyssinian_1.jpg"

with open(image_path, "rb") as f:
    response = requests.post(
        "http://localhost:3000/predict",
        files={
            "image": (
                "Abyssinian_1.jpg",
                f,
                "image/jpeg"
            )
        }
    )

print("Status code:", response.status_code)
print("Response:", response.text)