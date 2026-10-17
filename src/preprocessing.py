"""
The photo recipe: how every X-ray is prepared before the model sees it.

Used by: data_loader.py (and later predict.py)
"""

from torchvision import transforms

from src import config


def get_photo_recipe():
    """Return the steps that prepare one photo for the model."""
    photo_recipe = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # make every photo 224 x 224
        transforms.ToTensor(),                                      # turn the photo into numbers (0 to 1)
        transforms.Normalize(config.MEAN, config.STD),              # make the numbers small, near 0
    ])
    return photo_recipe
