from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet(
    root="./data/raw",
    split="trainval",
    target_types="category",
    download=False
)

print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))

print("\nClasses:")
for i, class_name in enumerate(dataset.classes):
    print(i, class_name)

image, label = dataset[0]

print("\nFirst image:")
print("Image type:", type(image))
print("Image size:", image.size)
print("Label:", label)
print("Breed:", dataset.classes[label])