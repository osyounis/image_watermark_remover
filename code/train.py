"""
This module is the training module used to train the Pix2Pix model.
"""
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import secret
import settings
import utilities
from dataset import WatermarkDataset
from generator_model import Generator
from discriminator_model import Discriminator



########################################
#            File Settings             #
########################################
torch.backends.cudnn.benchmark = True



########################################
#              Functions               #
########################################
def train_function(discriminator: object,
                   generator: object,
                   loader: object,
                   discriminator_optimizer: object,
                   generator_optimizer: object,
                   l1_loss: object,
                   bce: object,
                   generator_scaler: float,
                   discriminator_scaler: float
) -> None:
    """The training function which trains the Pix2Pix model on the DataSet.

    Args:
        discriminator (object): The discriminator model. 
        generator (object): The generator model.
        loader (object): The loader containing the dataset.
        discriminator_optimizer (object): The discriminator's optimizer.
        generator_optimizer (object): The generator optimizer.
        l1_loss (object): The L1-Loss object.
        bce (object): The Binary Cross Entropy.
        generator_scaler (float): The generator's scaler.
        discriminator_scaler (float): The discriminator's scaler.
    """
    loop = tqdm(loader, leave=True)
    
    for index, (input_image, target_image) in enumerate(loop):
        input_image = input_image.to(settings.DEVICE)
        target_image = target_image.to(settings.DEVICE)
        
        # Training the Discriminator
        with torch.cuda.amp.autocast():
            generated_image = generator(input_image)
            disc_input_image = discriminator(input_image, target_image)
            disc_input_image_loss = bce(disc_input_image, torch.ones_like(disc_input_image))
            
            disc_generated_image = discriminator(input_image, generated_image.detach())  
            disc_generated_image_loss = bce(disc_generated_image, torch.zeros_like(disc_generated_image))
            disc_loss = (disc_input_image_loss + disc_generated_image_loss) / 2
        
        discriminator.zero_grad()
        discriminator_scaler.scale(disc_loss).backward()
        discriminator_scaler.step(discriminator_optimizer)
        discriminator_scaler.update()
        
        # Training the Generator
        with torch.cuda.amp.autocast():
            disc_generated_image = discriminator(input_image, generated_image)
            generator_generated_image_loss = bce(disc_generated_image, torch.ones_like(disc_generated_image))
            L1 = l1_loss(generated_image, target_image) * settings.L1_LAMBDA
            generator_loss = generator_generated_image_loss + L1
        
        generator_optimizer.zero_grad()
        generator_scaler.scale(generator_loss).backward()
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()
        
        if index % 10 == 0:
            loop.set_postfix(
                disc_input_image = torch.sigmoid(disc_input_image).mean().item(),
                disc_generated_image = torch.sigmoid(disc_generated_image).mean().item(),
            )
        


def main() -> None:
    
    # Getting Start Time and sending message.
    start_time = datetime.now()
    start_time = start_time.strftime("%m/%d/%Y, %H:%M:%S")
    utilities.send_text_alert(f"Model training has started at {start_time}.", secret.TO_EMAIL_ADDRESS)
    
    # Training start
    discriminator = Discriminator(in_channels=settings.CHANNELS_IMG).to(settings.DEVICE)
    generator = Generator(in_channels=settings.CHANNELS_IMG, features=64).to(settings.DEVICE)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=settings.LEARNING_RATE, betas=(0.5, 0.999),)
    generator_optimizer = optim.Adam(generator.parameters(), lr=settings.LEARNING_RATE, betas=(0.5, 0.999),)
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    if settings.LOAD_MODEL:
        utilities.load_checkpoint(settings.CHECKPOINT_GEN, generator, generator_optimizer, settings.LEARNING_RATE)
        utilities.load_checkpoint(settings.CHECKPOINT_DISC, discriminator, discriminator_optimizer, settings.LEARNING_RATE)
    
    # Loading the training data.
    train_dataset = WatermarkDataset(root_directory=settings.TRAINING_DATA_DIRECTORY)
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS)
    
    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()
    
    # Loading the validation data.
    val_dataset = WatermarkDataset(root_directory=settings.VAL_DATA_DIRECTORY)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Training Loop.
    for epoch in range(settings.NUM_EPOCHS):
        train_function(discriminator,
                       generator,
                       train_loader,
                       discriminator_optimizer,
                       generator_optimizer,
                       L1_LOSS,
                       BCE,
                       generator_scaler,
                       discriminator_scaler)
        
        if settings.SAVE_MODEL and epoch % 5 == 0:
            utilities.save_checkpoint(generator, generator_optimizer, filename=settings.CHECKPOINT_GEN)
            utilities.save_checkpoint(discriminator, discriminator_optimizer, filename=settings.CHECKPOINT_DISC)
        
        utilities.save_examples(generator, val_loader, epoch, folder=settings.EVALUATION_DIRECTORY)

    # Getting End Time and sending message.
    end_time = datetime.now()
    end_time = end_time.strftime("%m/%d/%Y, %H:%M:%S")
    utilities.send_text_alert(f"Model training complete at {end_time}.", secret.TO_EMAIL_ADDRESS)

########################################
#              Main Code               #
########################################
if __name__ == "__main__":
    main()