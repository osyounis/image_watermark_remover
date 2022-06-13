"""
This module contains some utility functions used to save and load checkpoints.
It also has a function to save some example results from the model that is
being trained. Images will have an epoch number in their name to show when the
example was taken.
"""
import smtplib
from email.message import EmailMessage
import torch
from torchvision.utils import save_image

import settings
import secret



########################################
#              Functions               #
########################################
def save_checkpoint(model: object, optimizer: object, filename: str) -> None:
    """Save a checkpoint while training the model.

    Args:
        model (object): The model being trained.
        optimizer (object): The optimizer used during model training.
        filename (str): The path and filename of you want to save the
                        checkpoint to.
    """
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    
    torch.save(checkpoint, filename)



def load_checkpoint(checkpoint_file: str,
                    model: object,
                    optimizer: object,
                    learning_rate: float
) -> None:
    """Loads a previous checkpoint during training.

    Args:
        checkpoint_file (str): The path to where the checkpoint file was saved.
        model (object): The model that is being trained.
        optimizer (object): The optimizer used during model training.
        learning_rate (float): The initial learning rate used in model training.
    """
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=settings.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # This clears the previously learning rate for the previous checkpoint.
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate



def save_examples(generator: object,
                  val_loader: object,
                  epoch: int,
                  folder: str
) -> None:
    """Saves some image example as the model is trained. This includes input
    images, generated images and target images.

    Args:
        generator (object): The image generator model.
        val_loader (object): The validation loader.
        epoch (int): The current epoch being trained.
        folder (str): The path where the example images will be saved to.
    """
    input_image, target_image = next(iter(val_loader))
    input_image = input_image.to(settings.DEVICE)
    target_image = target_image.to(settings.DEVICE)
    generator.eval()
    
    with torch.no_grad():
        generated_image = generator(input_image)
        generated_image = generated_image * 0.5 + 0.5       # Removes normalization.
        save_image(generated_image, folder + f"/generated_image_{epoch}.png")
        save_image(input_image * 0.5 + 0.5, folder + f"/input_image_{epoch}.png")
        
        if epoch == 1:
            save_image(target_image * 0.5 + 0.5, folder + f"/target_image_{epoch}.png")
    
    generator.train()


def send_text_alert(message: str, to: str):
    """Sends a text message to a number.

    Args:
        message (str): The body of the message.
        to (str): The number followed by the correct SMS gateway.
    """
    
    # Setting up message.
    msg = EmailMessage()
    msg.set_content(message)
    msg['to'] = to
    msg['from'] = secret.ALERT_EMAIL
    
    # Sending message.
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(secret.ALERT_EMAIL, secret.APP_PASSWORD)
    server.send_message(msg)
    server.quit()