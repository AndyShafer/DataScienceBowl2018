# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:57:08 2018

@author: Andy
"""

import os
import pandas as pd
import imageio
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.io import imread, imshow, imread_collection, concatenate_images,show
from skimage.color import rgb2gray

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


def analyze_image(im_path, im_id):
    img = imread(im_path)[:,:,:3]
    im_gray = rgb2gray(img)
    
    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list, test_ids):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for im_path, im_id in zip(im_path_list, test_ids):
        im_df = analyze_image(im_path, im_id)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df


TEST_PATH = './input/test/'
test_ids = next(os.walk(TEST_PATH))[1]
testing = [TEST_PATH + test_id + '/images/' + test_id + '.png' for test_id in test_ids]
df = analyze_list_of_images(testing, test_ids)
df.to_csv('submission.csv', index=None)