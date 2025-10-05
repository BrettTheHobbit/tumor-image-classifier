#preprocessing the data so it may be used for the main classifier
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (224,224)
# Organizes all image data to be consistent across the whole dataset. Required to feed consistent data into the model
def preprocess():
    # Load dataset
    data_dir = Path("Brain_Tumor_Detection/train")
    pred_dir = Path("Brain_Tumor_Detection/pred")

    #resize all images (including prediction images)

    #split images into train and validation sets

    """ opening a single image
    yes = list(data_dir.glob('yes/*'))
    img = PIL.Image.open(str(yes[6]))
    img.show()
    """

    print("done preprocessing!") # Finished the preprocessing step!

#preprocess() # run the image preprocessing!


