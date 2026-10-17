"""
Check the X-ray images and make a clean copy.

The dataset is already split into Training, Validation and Testing,
so we keep that split as it is.

Steps:
  1. Find all images in the 3 original folders
  2. Remove empty and broken images
  3. Remove duplicate images (the same picture saved 2 or more times)
  4. Copy the good images to data/processed/
  5. Count the images and save a report in outputs/logs/

The original folder (Chest_X-ray) is never changed.

Run it from the chest-xray-pneumonia folder:
    python -m src.data_check
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the "src" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import shutil

from PIL import Image

from src import config

IMAGE_TYPES = [".png", ".jpg", ".jpeg"]


# ============================================================
# STEP 1: Find all images in the 3 original folders
# ============================================================
def list_images():
    """Return a list of (split, path, disease) for every image."""
    images = []
    for split in config.SPLITS:                     # Training, Validation, Testing
        for disease in config.CLASS_NAMES:          # Covid-19, Normal, ...
            folder = config.RAW_DIR / split / disease
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in IMAGE_TYPES:
                    images.append((split, path, disease))
    return images


# ============================================================
# STEP 2: Remove empty and broken images
# ============================================================
def is_good_image(path):
    """Return True if the file has data and opens as an image."""
    # Empty file? (0 bytes = no picture inside)
    if path.stat().st_size == 0:
        return False

    # Does it open as an image?
    try:
        with Image.open(path) as img:
            img.verify()  # checks that the file is not damaged
        return True
    except Exception:
        return False


def remove_bad_images(images):
    """Split the images into good ones and bad ones."""
    good = []
    bad = []
    for split, path, disease in images:
        if is_good_image(path):
            good.append((split, path, disease))
        else:
            bad.append((split, path, disease))
    return good, bad


# ============================================================
# STEP 3: Remove duplicate images
# ============================================================
def fingerprint(path):
    """Same picture -> same fingerprint. Used to find duplicates."""
    file_bytes = path.read_bytes()
    return hashlib.md5(file_bytes).hexdigest()


def remove_duplicates(images):
    """
    Keep only one copy of each picture.
    - Same picture with the SAME disease: keep the first copy.
      Training comes first, so if a picture is in Training AND Testing,
      it stays in Training and is removed from Testing (a fair test).
    - Same picture with DIFFERENT diseases: we don't know the true label,
      so we remove all copies.
    """
    # Put copies of the same picture together in a group
    groups = {}  # fingerprint -> list of (split, path, disease)
    for split, path, disease in images:
        key = fingerprint(path)
        if key not in groups:
            groups[key] = []
        groups[key].append((split, path, disease))

    keep = []
    removed = []

    for copies in groups.values():
        diseases = set()
        for split, path, disease in copies:
            diseases.add(disease)

        if len(diseases) > 1:
            removed.extend(copies)       # label conflict: remove all copies
        else:
            keep.append(copies[0])       # keep the first copy
            removed.extend(copies[1:])   # remove the other copies

    return keep, removed


# ============================================================
# STEP 4: Copy the good images to data/processed/
# ============================================================
def copy_images(images):
    """Copy every image to data/processed/<split>/<disease>/."""
    # Delete the old clean copy first, so we start fresh
    for split in config.SPLITS:
        old_folder = config.CLEAN_DIR / split
        if old_folder.exists():
            shutil.rmtree(old_folder)

    print(f"Copying {len(images)} images to {config.CLEAN_DIR} ...")
    for split, path, disease in images:
        new_folder = config.CLEAN_DIR / split / disease
        new_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, new_folder / path.name)


# ============================================================
# STEP 5: Count the images and save a report
# ============================================================
def count_images(images):
    """Count images for each split and disease."""
    counts = {}
    for split in config.SPLITS:
        counts[split] = {}
        for disease in config.CLASS_NAMES:
            counts[split][disease] = 0

    for split, path, disease in images:
        counts[split][disease] += 1

    return counts


def write_report(counts, bad, duplicates):
    """Print the report and save it to outputs/logs/."""
    lines = []
    lines.append("DATA CHECK REPORT")
    lines.append("=================")
    lines.append("Source: the original Training, Validation and Testing folders")
    lines.append("")
    lines.append(f"Empty or broken images removed: {len(bad)}")
    lines.append(f"Duplicate images removed:       {len(duplicates)}")
    lines.append("")
    lines.append("Images in each part:")

    for split in config.SPLITS:
        lines.append(f"  {split}")
        for disease in config.CLASS_NAMES:
            number = counts[split][disease]
            lines.append(f"    {disease:<14} {number:>5}")

    report = "\n".join(lines)
    print()
    print(report)

    # Save the report and the list of removed files
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOGS_DIR / "data_report.txt").write_text(report + "\n")

    removed_lines = []
    for split, path, disease in bad:
        removed_lines.append(f"empty or broken  {path}")
    for split, path, disease in duplicates:
        removed_lines.append(f"duplicate        {path}")
    (config.LOGS_DIR / "removed_files.txt").write_text("\n".join(removed_lines) + "\n")

    print()
    print(f"Saved: {config.LOGS_DIR / 'data_report.txt'}")
    print(f"Saved: {config.LOGS_DIR / 'removed_files.txt'}")


# ============================================================
# Run every step, one after the other
# ============================================================
def run_all():
    print("Step 1: finding images ...")
    images = list_images()

    print("Step 2: removing empty and broken images ...")
    good, bad = remove_bad_images(images)

    print("Step 3: removing duplicates ...")
    unique, duplicates = remove_duplicates(good)

    print("Step 4: copying ...")
    copy_images(unique)

    print("Step 5: counting ...")
    counts = count_images(unique)
    write_report(counts, bad, duplicates)
    return counts


if __name__ == "__main__":
    run_all()
