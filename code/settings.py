"""
This module contains all settings/configurations used in this project.
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Directories and File Locations.
ORIGINAL_IMAGE_DIRECTORY = '../data/original_images'
WATERMARKED_IMAGE_DIRECTORY = '../data/watermarked_images'
TRAINING_DATA_DIRECTORY = '../data/train_data'
VAL_DATA_DIRECTORY = "../data/val_data"
CHECKPOINT_DIRECTORY = "./checkpoints"
EVALUATION_DIRECTORY = "../evaluation"
CHECKPOINT_DISC = "./checkpoints/disc.pth.tar"
CHECKPOINT_GEN = "./checkpoints/gen.pth.tar"

# Image Download Settings
TOTAL_NUM_OF_IMAGES = 16_707

# Training Device Configurations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Configurations
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 4
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10

# Model Save/Load Configurations
LOAD_MODEL = False
SAVE_MODEL = True

# Image Augmentations
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [A.HorizontalFlip(p=0.5),
     A.ColorJitter(p=0.2),
     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
     ToTensorV2(),]
)

transform_only_mask = A.Compose(
    [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
     ToTensorV2()]
)