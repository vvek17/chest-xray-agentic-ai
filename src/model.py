"""
The model (the "brain").

We use a model that already learned from 1 million normal photos (ImageNet).
We change only its LAST layer, so it gives 3 answers (our 3 diseases).

Which model: set MODEL_NAME in config.py ("resnet18" or "densenet121").

Used by: train.py and evaluate.py
"""

import torch.nn as nn
from torchvision import models

from src import config


def build_model(pretrained=True):
    """
    pretrained=True  -> download the ImageNet knowledge (for training)
    pretrained=False -> empty model; we fill it later from our saved file (for testing)
    """
    if pretrained:
        weights = "DEFAULT"
    else:
        weights = None

    if config.MODEL_NAME == "resnet18":
        model = models.resnet18(weights=weights)
        number_of_inputs = model.fc.in_features                     # how many numbers go into the last layer
        model.fc = nn.Linear(number_of_inputs, config.NUM_CLASSES)  # new last layer: 3 diseases

    elif config.MODEL_NAME == "densenet121":
        model = models.densenet121(weights=weights)
        # In DenseNet, the last layer is called "classifier" (in ResNet it is "fc")
        number_of_inputs = model.classifier.in_features
        model.classifier = nn.Linear(number_of_inputs, config.NUM_CLASSES)

    else:
        raise ValueError(f"Unknown MODEL_NAME in config.py: {config.MODEL_NAME}")

    return model
