"""
Clean any chest X-ray photo before it is classified.

Photos from outside our dataset (screenshots, photos from the internet, other hospitals)
can look different: color instead of gray, empty borders, too dark or too bright.
This step makes them look more like the X-rays in our dataset.

  Step 1: make the photo gray
  Step 2: cut off empty borders (plain black or white frames)

We tested a 3rd step, "fix brightness and contrast" (autocontrast), on 100 of our
Testing photos: it LOWERED accuracy from 97/100 to 91/100, so it is not used.
Steps 1 and 2 did not change accuracy (97/100).

Note: cleaning fixes the LOOK of the photo. It cannot fix bigger differences,
for example an adult X-ray when the model only saw children with pneumonia.

Try it on one photo from the chest-xray-pneumonia folder:
    python -m backend.clean_image path/to/xray.png
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageOps


def cut_empty_borders(gray_image):
    """Remove rows and columns at the edges that are almost one plain color."""
    pixels = np.array(gray_image).astype(float)

    # How much does each row / column change? A plain border changes almost nothing.
    row_change = pixels.std(axis=1)
    column_change = pixels.std(axis=0)

    useful_rows = np.where(row_change > 5)[0]
    useful_columns = np.where(column_change > 5)[0]

    # Nothing useful found: keep the photo as it is
    if len(useful_rows) == 0 or len(useful_columns) == 0:
        return gray_image

    top, bottom = useful_rows[0], useful_rows[-1] + 1
    left, right = useful_columns[0], useful_columns[-1] + 1

    # Safety: if this would cut away more than half of the photo, do not cut
    new_area = (bottom - top) * (right - left)
    old_area = pixels.shape[0] * pixels.shape[1]
    if new_area < 0.5 * old_area:
        return gray_image

    return gray_image.crop((left, top, right, bottom))


def clean_xray(image):
    """
    image: a photo path, or an image that is already open.
    Returns the cleaned image (RGB, because the models expect 3 channels).
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Step 1: gray
    gray = ImageOps.grayscale(image)

    # Step 2: cut off empty borders
    gray = cut_empty_borders(gray)

    return gray.convert("RGB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.clean_image path/to/xray.png")
        sys.exit()

    original = Image.open(sys.argv[1])
    cleaned = clean_xray(original)
    save_path = Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "_cleaned.png")
    cleaned.save(save_path)
    print("Original size:", original.size, "| Cleaned size:", cleaned.size)
    print("Saved the cleaned photo:", save_path)
