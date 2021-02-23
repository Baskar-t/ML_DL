
#dataset.py

# Imports

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img , img_to_array , ImageDataGenerator

import initialize

#import train data
train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,horizontal_flip = True,
                                   vertical_flip =  True ,
                                   rotation_range=60)


train_data = train_datagen.flow_from_directory(initialize.DATA_PATH+'train',
                                                 target_size = (244, 244),
                                                 class_mode='sparse',
                                                 shuffle=True,seed=1)
#import val data

val_datagen = ImageDataGenerator(rescale = 1/255)
val_data = val_datagen.flow_from_directory(initialize.DATA_PATH+'val',
                                                           target_size=(244,244),
                                                           class_mode='sparse',
                                                           shuffle=True,seed=1)

# import test data


test_datagen = ImageDataGenerator(rescale = 1/255)
test_data = test_datagen.flow_from_directory(initialize.DATA_PATH1+'Test',
                                                           target_size=(244,244),
                                                           class_mode='sparse',
                                                           shuffle=False,seed=1)

