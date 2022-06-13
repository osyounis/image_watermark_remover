"""
This module collects a bunch of watermark free images from 
https://unsplash.com/. It then creates a copy with a watermark and combines the
two images into a single image. This single images is the data that is used for
training. 

NOTE: This method only supplies 30 images per search due to the API's 
      restrictions. But the unsplash API allows for 5,000 requests per hour.

NOTE: Some of these functions are unused. As I worked on this project, I discovered
new information and techniques to make the code better and quicker. One example
of this is that originally I used to the Unsplash API to search and download 
images for my data. This was not very efficient and left me with big files for
every image. By using the Unsplash LITE dataset, I was able to collect more images
faster and could use the dynamic link to get smaller image files.
"""

import os
import shutil
import requests
import random
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import settings
import secret
from utilities import send_text_alert



########################################
#              Functions               #
########################################
def check_directory(directory: str) -> None:
    """Checks if a directory exists. If it does not exist, this function creates
    that directory.

    Args:
        directory (str): The path to the directory that is being checked. 
        Must be in str format
    """
    current_directory = os.getcwd()
    output_directory = current_directory + '/' + directory
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)


def collect_images_from_dataset(photo_url: str, filename: str) -> None:
    """Uses the Unsplash Dataset to collect images. In the dataset contains
    dynamic links to each image. These images are then cropped to 600 X 600 to
    make them smaller. This makes them quicker to download and makes them the
    same size.

    Args:
        photo_url (str): The dynamic url for an image from the Unsplash dataset.
        filename (str): The filename the image will be saved as.
    """
    try:
        image_bytes = requests.get(f"{photo_url}?fit=crop&w=600&h=600", timeout=30)
    
    except requests.exceptions.ConnectTimeout:
        with open("errors.txt", 'a') as f_obj:
            f_obj.write(f"{filename}, CONNECTION TIMEOUT, Unable to connect to image\n")
        
    except requests.exceptions.ReadTimeout:
        with open("errors.txt", 'a') as f_obj:
            f_obj.write(f"{filename}, RESPONSE TIMEOUT, Unable to download image\n")
        
    # print(image_bytes.status_code)
    image_bytes = image_bytes.content
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    
    image.save(settings.ORIGINAL_IMAGE_DIRECTORY + '/' + filename + '.png')



def create_image_train_data(input_file: str, watermark_text: str, font_size: 10):
    """Takes the original image and creates a new one. The new image is a combined
    image of the original image and a watermarked version of the image.
    The original image is on the left side and the watermarked version is on the
    right. This new image is then saved to the training data directory.

    Args:
        input_file (str): The name of the original image.
        watermark_text (str): The text that will be used to created the watermark.
        font_size (10): The font size of the watermark.
    """
    try:
        # Opening Original Image
        image = Image.open(settings.ORIGINAL_IMAGE_DIRECTORY + "/" + input_file).convert('RGBA')
        text_obj = Image.new('RGBA', image.size, (255, 255, 255, 0))
        font = ImageFont.truetype('arial.ttf', font_size)
        
        # Create watermark layer.
        draw_obj = ImageDraw.Draw(text_obj)
        
        # Adding watermarks to the watermark layer.
        width, height = image.size
        
        y = 200
        for i in range(10):
            x = random.randint(0, width)
            y += random.randrange(0, int(height / 8), 19) + random.randint(0, 100)
            draw_obj.text((x, y), watermark_text, fill=(255, 255, 255, 175), font=font)
            
        # Combining the original image layer with the watermark layer.
        watermarked_image = Image.alpha_composite(image, text_obj)
        
        # Creating new image object and pasting both images in it.
        new_image = Image.new('RGBA', (image.width + watermarked_image.width, image.height))
        new_image.paste(image, (0, 0))
        new_image.paste(watermarked_image, (image.width, 0))
        
        # Saving new image.
        new_image.save(settings.TRAINING_DATA_DIRECTORY + "/" + input_file)
    
    except:
        with open("errors.txt", 'a') as f_obj:
            f_obj.write(f"{input_file}, IMAGE CREATION ERROR, Unable to create train data image\n")



def collect_images(query: str) -> None:
    """THIS FUNCTION IS NO LONGER NEEDED.
    
    Takes a search term and downloads the first 30 images for from the
    unsplash API search results. The images are saved in the directory assigned
    to the 'ORIGINAL_IMAGE_DIRECTORY' variable found in the 'settings.py' file. 

    Args:
        query (str): A keyword used to search images on unsplash, e.g. san diego.
    """
    # Cleans up query string to get it ready for url link.
    query = query.strip().replace(" ", "-")
    
    # The url needed to send searches queries using the unsplash API.
    api_url = f'https://api.unsplash.com/search/photos/?per_page=30&query={query}&client_id={secret.CLIENT_ID}'
    
    # Getting the json data for the returned search results.
    r = requests.get(api_url)
    json_data = r.json()
    
    # Finding the the image urls in the json and giving them a name to save.
    for index, image in enumerate(json_data['results']):
        image_url = image['urls']['raw']
        image_name = f"{query}_original_{index}.png"
        
        # Saving the image.
        try:
            with open(settings.ORIGINAL_IMAGE_DIRECTORY + "/" + image_name, 'wb') as f_obj:
                r_image =  requests.get(image_url)
                f_obj.write(r_image.content)
        except:
            print("Could not save image.")

        

def create_watermarked_image(filename: str, text: str, font_size=100)-> None:
    """THIS FUNCTION IS NO LONGER NEEDED.
    
    Takes and non-watermarked image and creates a copy off it with a
    watermark. The new image is saved in the directory assigned to the 
    'WATERMARKED_IMAGE_DIRECTORY' variable found in the 'settings.py' file.

    Args:
        filename (str): The filename of the non-watermarked image.
        text (str): The text that will be in the watermark, e.g. shutterstock.
        font_size (int, optional): The font size of the text in the watermark. Defaults to 100.
    """
    # Creating the output filename.
    watermarked_filename = filename.replace("original", "watermarked")
    
    # Getting the original image.
    try:
        image = Image.open(settings.ORIGINAL_IMAGE_DIRECTORY + "/" + filename).convert('RGBA')
        text_obj = Image.new('RGBA', image.size, (255, 255, 255, 0))
        font = ImageFont.truetype('arial.ttf', font_size)
        
        # Create watermark layer.
        draw_obj = ImageDraw.Draw(text_obj)
        
        # Adding watermarks to the watermark layer.
        width, height = image.size
        
        y = 200
        for i in range(10):
            x = random.randint(0, width)
            y += random.randrange(0, int(height / 8), 19) + random.randint(0, 100)
            draw_obj.text((x, y), text, fill=(255, 255, 255, 175), font=font)
            
        # Combining the original image layer with the watermark layer.
        watermarked_image = Image.alpha_composite(image, text_obj)
        watermarked_image.save(settings.WATERMARKED_IMAGE_DIRECTORY + "/" + watermarked_filename)
    except:
        pass



def combine_images(filename: str) -> None:
    """THIS FUNCTION IS NO LONGER NEEDED.
    
    Takes two images and combines them into one image. Each image in the new
    image are side by side (along the horizontal). The new image is saved in the
    directory assigned to the 'TRAINING_DATA_DIRECTORY' variable found in the 
    'settings.py' file.

    Args:
        filename (str): The filename of the watermarked image.
    """
    # Create output filename.
    original_filename = filename.replace("watermarked", "original")
    watermarked_filename = filename
    output_filename = original_filename.replace("original_", "")
    
    try:
        # Opening the two images to combine.
        image_1 = Image.open(settings.ORIGINAL_IMAGE_DIRECTORY + "/" + original_filename)
        image_2 = Image.open(settings.WATERMARKED_IMAGE_DIRECTORY + "/" + watermarked_filename)
        
        # Creating new image and pasting both images in it.
        new_image = Image.new('RGBA', (image_1.width + image_2.width, image_1.height))
        new_image.paste(image_1, (0, 0))
        new_image.paste(image_2, (image_1.width, 0))
        
        # Saving new image.
        new_image.save(settings.TRAINING_DATA_DIRECTORY + "/" + output_filename)
    except:
        pass




########################################
#              Main Code               #
########################################
needed_directories = [settings.ORIGINAL_IMAGE_DIRECTORY,
                      settings.TRAINING_DATA_DIRECTORY,
                      settings.VAL_DATA_DIRECTORY,
                      settings.CHECKPOINT_DIRECTORY,
                      settings.EVALUATION_DIRECTORY]

watermark_texts = ["Watermark.com", "Resplash.com", "Younivate.com",
                   "Lonprox.com", "shutterstock", "FalconStocks", 
                   "Guardians.com", "Youniworks.com", "OtterBox.com", "TopGun"]

# Checking and making needed directories if needed.
for directory in tqdm(needed_directories, desc="Checking Directories"):
    check_directory(directory)
    
# Creating a DataFrame with the Unsplash Dataset with the information I need.
df = pd.read_csv(
    "../data/unsplash_dataset/photos.tsv000",
    sep="\t",
    header=0
)

df = df[['photo_id', 'photo_image_url']]

# Downloading original resized images.
for i in tqdm(range(settings.TOTAL_NUM_OF_IMAGES + 1), desc="Downloading original images", total=settings.TOTAL_NUM_OF_IMAGES + 1):
    try:
        image_url = df['photo_image_url'][i]
        image_name = df['photo_id'][i]
        collect_images_from_dataset(image_url, image_name)
        if i % 1000 == 0:
            time.sleep(5)
    except:
        with open("errors.txt", 'a') as f_obj:
            f_obj.write(f"{image_name}, UNKNOWN, Unable to download image\n")
        

# Sending Progress Update.
current_time = datetime.now()
current_time = current_time.strftime("%m/%d/%Y, %H:%M:%S")
send_text_alert(f"Image file have been downloaded. Now working on creating dataset images at {current_time}.", secret.TO_EMAIL_ADDRESS)

# Creating data images and saving them to the training file.
original_files = os.listdir(settings.ORIGINAL_IMAGE_DIRECTORY)

for image_file in tqdm(original_files, desc="Creating training images", total=len(original_files)):
    watermark_text =  random.choice(watermark_texts)
    create_image_train_data(image_file, watermark_text, font_size=20)

# Splitting up data into train and val data (80% train and 20% val split of data).
total_data = os.listdir(settings.TRAINING_DATA_DIRECTORY)
val_quantity = int(0.2 * len(total_data))
val_files = random.sample(total_data, val_quantity)

for val_file in tqdm(val_files, desc="Splitting data", total=len(val_files)):
    shutil.move(settings.TRAINING_DATA_DIRECTORY + "/" + val_file,
                settings.VAL_DATA_DIRECTORY + "/" + val_file)


# Sending Completion Update.
end_time = datetime.now()
end_time = end_time.strftime("%m/%d/%Y, %H:%M:%S")
send_text_alert(f"'data_collection.py' Completed at {end_time}! Dataset is ready for training.", secret.TO_EMAIL_ADDRESS)




# ################################################################################
# # Ths code below is my old method of creating the data for training.           #
# ################################################################################

# required_directories = [settings.ORIGINAL_IMAGE_DIRECTORY,
#                         settings.WATERMARKED_IMAGE_DIRECTORY,
#                         settings.TRAINING_DATA_DIRECTORY,
#                         settings.VAL_DATA_DIRECTORY,
#                         settings.CHECKPOINT_DIRECTORY]

# search_queries = ["surfing", "san diego", "food", "wildlife", "dog", "skyline",
#                   "bridge", "alaska", "nature", "marine life", "abstract",
#                   "technologies", "aerial images", "buildings", "education", 
#                   "holidays", "objects", "interior", "miscellaneous", "outdoors",
#                   "people", "seasons", "arabic", "weather", "coast guard",
#                   "us army", "sports", "transport", "boats", "computers",
#                   "medical", "electric", "smile", "windsurf", "fitness"]

# watermark_texts = ["Watermark.com", "Resplash.com", "Younivate.com",
#                    "Lonprox.com", "shutterstock", "FalconStocks", 
#                    "Guardians.com", "Youniworks.com", "OtterBox.com", "TopGun"]


# # Checking and making the required directories.
# for directory in tqdm(required_directories, desc="Checking Directories"):
#     check_directory(directory)


# # Getting images needed.
# for index, search_query in enumerate(tqdm(search_queries, desc="Downloading Images")):
#     collect_images(search_query)
    
#     # Adding delays to that the API doesn't receive requests too fast. 
#     if index % 10 == 0:
#         time.sleep(30)
#     else:
#         time.sleep(1)


# # Getting a list of all original images.
# original_image_files = os.listdir(settings.ORIGINAL_IMAGE_DIRECTORY)

# # Creating watermarked image files.
# for image_file in tqdm(original_image_files, desc="Adding Watermarks"):
#     watermark_text = random.choice(watermark_texts)
#     create_watermarked_image(image_file, watermark_text)

# # Getting a list of all watermarked images.
# watermarked_image_files = os.listdir(settings.WATERMARKED_IMAGE_DIRECTORY)

# # Creating data image files.
# for image_file in tqdm(watermarked_image_files, desc="Combining Images"):
#     combine_images(image_file)


# # Split up data into train and val data.
# total_data = os.listdir(settings.TRAINING_DATA_DIRECTORY)

# # Creating 80%-20% split of data and moving the 20% to the val folder.
# val_quantity = int(0.2 * len(total_data))
# val_data = random.sample(total_data, val_quantity)

# for val_file in tqdm(val_data, desc="Splitting Data"):
#     shutil.move(settings.TRAINING_DATA_DIRECTORY + "/" + val_file,
#                 settings.VAL_DATA_DIRECTORY + "/" + val_file)