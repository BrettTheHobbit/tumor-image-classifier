#preprocessing the data so it may be used for the main classifier
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (224,224)
BATCH_SIZE = 32
DATASET_SPLIT = 0.2

# Organizes all image data to be consistent across the whole dataset. Required to feed consistent data into the model
def preprocess():
    # Load dataset
    # Get the folder where script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    #find the full path for both test and training data folders 
    data_dir = os.path.join(script_dir, "Brain_Tumor_Detection", "train")
    test_dir  = os.path.join(script_dir, "Brain_Tumor_Detection", "test")

    # Clean up path formatting
    data_dir = os.path.normpath(data_dir)
    test_dir  = os.path.normpath(test_dir)

    #resize and split images into test/validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = DATASET_SPLIT,
        subset = "training",
        seed = 20251005,
        image_size =(IMG_SIZE[0],IMG_SIZE[1]),
        batch_size = BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = DATASET_SPLIT,
        subset = "training",
        seed = 20251005,
        image_size =(IMG_SIZE[0],IMG_SIZE[1]),
        batch_size = BATCH_SIZE
    )

    #changing image sizes of test data for consistentcy
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,   
        batch_size=BATCH_SIZE,
        shuffle=False,    
        label_mode=None      
    )   
    #normalize all datasets (could do it in pipeline)
    normalize = lambda x, y: (x / 255.0, y)

    # labelled normalization
    train_ds = train_ds.map(normalize)
    val_ds = val_ds.map(normalize)

    # unlabelled normalization
    test_ds = test_ds.map(lambda x: x / 255.0)

    
    

    print("done preprocessing!") # Finished the preprocessing step!

def show_batch():
    pass

#used for debugging/sanity checks
def open_img(folder_path, img_index, folder_label):
    #opening a single image
    data_dir_path = Path(folder_path) #specifically needs to be a path object to display the image
    label = list(data_dir_path.glob(folder_label + '/*'))
    img = PIL.Image.open(str(label[img_index]))
    img.show()

preprocess() # run the image preprocessing!


