import json
import numpy as np
from PIL import Image
from pathlib import Path

DATA_DIR = Path("data/raw/oxford-iiit-pet/images")

REFERENCE_SIZE = 500
CURRENT_SIZE = 500
BINS = 10
THRESHOLD = 0.25


def get_brightness_values(image_paths):
    values = []

    for path in image_paths:
        image = Image.open(path).convert("L")
        image = np.array(image)

        values.append(image.mean())

    return np.array(values)


images = sorted(DATA_DIR.glob("*.jpg"))

reference_images = images[:REFERENCE_SIZE]
current_images = images[-CURRENT_SIZE:]

reference = get_brightness_values(reference_images)
current = get_brightness_values(current_images)


def calculate_psi(reference, current, bins=10):

    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


psi = calculate_psi(reference, current)

status = "DRIFT DETECTED" if psi > THRESHOLD else "NO SIGNIFICANT DRIFT"

print(f"Reference images: {len(reference)}")
print(f"Current images: {len(current)}")
print(f"PSI: {psi:.4f}")
print(f"Threshold: {THRESHOLD}")
print(f"Status: {status}")


with open("monitoring_metrics.json", "w") as f:
    json.dump(
        {
            "psi": psi,
            "threshold": THRESHOLD,
            "status": status
        },
        f,
        indent=2
    )