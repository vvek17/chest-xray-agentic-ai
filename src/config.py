from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original data (complete dataset, 15,600 images; we never change it)
RAW_DIR = PROJECT_ROOT.parent / "Chest_X-ray"

# Clean copy made by src/data_check.py (the model uses this one)
CLEAN_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["Training", "Validation", "Testing"]
TRAIN_DIR = CLEAN_DIR / "Training"
VAL_DIR = CLEAN_DIR / "Validation"
TEST_DIR = CLEAN_DIR / "Testing"

MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

# Which pretrained model to use: "resnet18" or "densenet121"
MODEL_NAME = "densenet121"

# Each model is saved in its own file, e.g. models/best_densenet121.pth
BEST_MODEL_PATH = MODEL_DIR / f"best_{MODEL_NAME}.pth"

# Frozen pretrained model (no fine-tuning), used by src/dinov2_classify.py
#   "facebook/dinov2-base"  = DINOv2, learned from normal photos
#   "microsoft/rad-dino"    = RAD-DINO, a DINOv2 that learned from chest X-rays
#   "google/medsiglip-448"  = MedSigLIP, learned from medical images AND text
#                             (it can also do zero-shot; needs Hugging Face login)
DINOV2_NAME = "google/medsiglip-448"            # downloaded from Hugging Face

# Zero-shot (MedSigLIP only): the sentence used for each disease
ZERO_SHOT_PROMPTS = {
    "Covid-19": "a chest x-ray showing covid-19 pneumonia",
    "Normal": "a normal chest x-ray",
    "Pneumonia": "a chest x-ray showing pneumonia",
    "Tuberculosis": "a chest x-ray showing tuberculosis",
}
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"   # saved DINOv2 embeddings

# The 4 diseases (folder names, in A-B-C order)
CLASS_NAMES = ["Covid-19", "Normal", "Pneumonia", "Tuberculosis"]
NUM_CLASSES = len(CLASS_NAMES)

# The dataset is already split into Training, Validation and Testing folders,
# so we use those folders as they are.

# Data Augmentation
BRIGHTNESS = (0.64, 1.37) # (MIN, MAX)
CONTRAST   = (0.64, 1.37) # (MIN, MAX)


# Model Training
IMAGE_SIZE = 224
# MEAN = the average color of all those photos
# Red: 0.485, Green: 0.456, Blue: 0.406
# STD = how much the colors change from photo to photo
# Red: 0.229, Green: 0.224, Blue: 0.225
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42
NUM_WORKERS = 2

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
