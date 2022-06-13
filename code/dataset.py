"""
The module contains the classes used to create the dataset for training the 
model.
"""

import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import settings


class WatermarkDataset(Dataset):
    def __init__(self, root_directory) -> None:
        """Inits WatermarkDataset with a directory."""
        self.root_directory = root_directory
        self.list_files = os.listdir(self.root_directory)
    
    def __len__(self) -> int:
        """Returns the number of files in the dataset."""
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> tuple:
        """Takes and image and splits it into two. One is the input image, 
        the other is the target image.

        Args:
            index (int): The index of the current image from the image list.

        Returns:
            tuple: Two arrays in a tuple. The first is the input image and the 
                   second is target image.
        """
        image_file = self.list_files[index]
        image_path = os.path.join(self.root_directory, image_file)
        image_object = Image.open(image_path).convert('RGB')
        # image_width = image_object.width
        image = np.array(image_object)
        
        # The center point width-wise to split the image in two.
        # center_width = int(image_width / 2)
        center_width = int(image.shape[1] / 2)
        
        # Splitting image in two. Input image and Target image.
        input_image = image[:, center_width:, :]
        target_image = image[:, :center_width, :]
        
        # Adding Augmentations to the images.
        augmentations = settings.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]
        
        input_image = settings.transform_only_input(image=input_image)["image"]
        target_image = settings.transform_only_mask(image=target_image)["image"]
        
        return input_image, target_image
    
    
if __name__ == "__main__":
    dataset = WatermarkDataset(settings.TRAINING_DATA_DIRECTORY + "/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y, in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        
        sys.exit()