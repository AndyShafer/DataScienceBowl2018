# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 00:57:24 2018
@author: mason
"""
# Imports
import tensorflow as tf
import keras
import os
import sys
import numpy as np
import random
import math
import warnings
import pandas as pd
import sklearn
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images,show
from skimage.transform import resize
from skimage.morphology import label
from itertools import chain
from cnn_model import cnn_model_fn

#%%
# Set randomness so repeatable
seed = 42
random.seed = seed
np.random.seed = seed

#%%
# Set some Image importing parameters
IMG_WIDTH = 350
IMG_HEIGHT = 350
IMG_CHANNELS = 3 #RGB
IMG_MAX = 256
TRAIN_PATH = './input/train/'
TEST_PATH = './input/test/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#%%
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

#%%
# Allocate memory for images and lables, note lables are bool, images are 8 bit
images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
labels = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

#%%
# Get images and lables, and resize to parameters
for i in range(len(train_ids)):
#for i in range(5):
     
    #this is the image ID
    id_ = train_ids[i]
    #path to image based on ID
    path = TRAIN_PATH + id_
    #read image into img
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    #resize image
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #rescale image
    img_max = img.max()
    img_min = img.min()
    img = (((img - img_min)/(img_max - img_min))*IMG_MAX)
    #save image
    images[i] = img[:,:,:]
    
    #get the mask == lable
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)  
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    #save lable
    labels[i] = mask

#%%
# Set ML Hyperparameters, isolate data
#training data section
train_size = int(0.9 * len(train_ids))

# Training Data
X_train = images[:train_size,:,:,:]
# Training Lables
Y_train = labels[:train_size,:,:,0]
Y_train = Y_train.astype(np.float32)

# Validation Data
X_validate = images[train_size:,:,:,:]
# Validation Lables
Y_validate = labels[train_size:,:,:,0]
Y_validate = Y_validate.astype(np.float32)

#%%
# Look at some images
for i in range(5):
    rand_idx = random.randint(0,train_size)
    imshow(X_train[rand_idx])
    show()
    imshow(Y_train[rand_idx])
    show()

estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model="/models/cnn_1")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=Y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
estimator.train(
        input_fn=train_input_fn,
        steps=1000)

eval_input_fn = tf.extimator.inputs.numpy_input_fn(
        x={"x": X_validate},
        y=Y_validate,
        num_epochs=1,
        shuffle=False)

test_results = estimator.evaluate(input_fn=eval_input_fn)