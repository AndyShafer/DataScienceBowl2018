# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:48:55 2018

@author: Andy
"""

import tensorflow as tf
import keras
import os
import sys
import csv
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
from itertools import chain
from scipy import ndimage

def rle_encoding(x):
    x = x.T.flatten()
    encoded_pixels = []
    pixel = 0
    while(pixel < len(x)):
        if(x[pixel] == 1):
            encoded_pixels.append(pixel+1)
            pixel_length = 1
            pixel += 1
            while(pixel < len(x) and x[pixel] == 1):
                pixel_length += 1
                pixel += 1
            encoded_pixels.append(pixel_length)
        pixel += 1
    return encoded_pixels
    
def display_prediction(prediction): 
    prediction = prediction.astype(np.float32)
    imshow(prediction)
    show()
    
def label_nuclei(prediction):
    labels, nlabels = ndimage.label(prediction)
    
    print('There are {} separate components / objects detected.'.format(nlabels))
    
    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
        cell = prediction[label_coords]
        # Check if the label is too small
        if(np.product(cell.shape) < 10):
            prediction = np.where(labels==label_ind+1, 0, prediction)
    
    # Regenerate labels
    labels, nlabels = ndimage.label(prediction)
    print("There are now {} separate components / objects detected.".format(nlabels))
    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    
    return prediction, label_arrays

def generate_output_file():
    IMG_CHANNELS = 3 #RGB
    TEST_PATH = './input/test/'
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    
    with open('submission.csv', 'w', newline='') as submission:
        writer = csv.writer(submission, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ImageId', 'EncodedPixels'])
    
        test_ids = next(os.walk(TEST_PATH))[1]
        print(len(test_ids))
        for i, test_id in enumerate(test_ids):
            path = TEST_PATH + test_id
            img = imread(path + '/images/' + test_id + '.png')[:,:,:IMG_CHANNELS]
            imshow(img)
            show()
        
            height = len(img)
            width = len(img[0])
            img = np.resize(img, (height * width, 3))
            kmeans = KMeans(n_clusters=2).fit(img)
            prediction = kmeans.labels_
            prediction = np.resize(prediction, (height, width))
            display_prediction(prediction)
            print('Enter "y" if the prediction needs to be inverted: ')
            if(input() == 'y'):
                prediction[prediction == 0] = 2
                prediction[prediction == 1] = 0
                prediction[prediction == 2] = 1
                display_prediction(prediction)
            
            prediction, label_arrays = label_nuclei(prediction)
            display_prediction(prediction)
            for label_mask in label_arrays:
                encoded_pixels = rle_encoding(label_mask)
                writer.writerow([test_id, ' '.join(map(str, encoded_pixels))])
            

generate_output_file()
'''
IMG_CHANNELS = 3 #RGB
IMG_MAX = 256
TRAIN_PATH = './input/train/'
TEST_PATH = './input/test/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#%%
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

id_ = train_ids[4]
path = TRAIN_PATH + id_
img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

#rescale image
img_max = img.max()
img_min = img.min()
#img = (((img - img_min)/(img_max - img_min))*IMG_MAX)

mask = np.zeros((len(img), len(img[0]), 1), dtype=bool)
for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (len(mask), len(mask[0])), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)

label = mask[:,:,0]
#label = label.astype(np.float32)

imshow(img)
show()
imshow(label)
show()

height = len(img)
width = len(img[0])
img = np.resize(img, (height * width, 3))
print(len(img))
kmeans = KMeans(n_clusters=2, ).fit(img)
print(len(kmeans.labels_))
prediction = np.resize(kmeans.labels_, (height, width))
prediction = prediction.astype(np.float32)
imshow(prediction)
show()

'''