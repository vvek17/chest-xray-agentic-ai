"""
Load the X-ray photos from the 3 folders and put them into batches.
Folder name = disease name (the "label").

Used by: train.py and evaluate.py
"""

from torch.utils.data import DataLoader
from torchvision import datasets

from src import config
from src.preprocessing import get_photo_recipe


def load_photos(folder):
    """Load all photos from one folder (Training, Validation, or Testing)."""
    photos = datasets.ImageFolder(folder, transform=get_photo_recipe())
    return photos


def make_batches(photos, shuffle):
    """
    Put the photos into groups of BATCH_SIZE (32).
    shuffle=True  -> mix the photos in a new order every epoch (for training)
    shuffle=False -> keep the same order (for checking)
    """
    batches = DataLoader(photos, batch_size=config.BATCH_SIZE, shuffle=shuffle)
    return batches
