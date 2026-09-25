import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet


dataset = OxfordIIITPet(
    root="./data/raw",
    split="trainval",
    target_types="category",
    download=False
)


fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    image, label = dataset[i]

    ax.imshow(image)
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.tight_layout()
plt.show()