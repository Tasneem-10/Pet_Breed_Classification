from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet(
    root="./data/raw",
    split="trainval",
    target_types="category",
    download=True
)

print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("First class:", dataset.classes[0])

