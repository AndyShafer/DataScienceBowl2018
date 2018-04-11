# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:18:15 2018

@author: Andy
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from scipy import ndimage

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/train/'
TEST_PATH = './input/test/'
AUGMENTATIONS_PER_IMG = 4

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed(seed=seed)

# Create the generator to augment the data
def create_data_generator(X_train, Y_train):
    datagen = ImageDataGenerator(rotation_range=45., width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, vertical_flip=True, fill_mode='reflect')
    #datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    datagen.fit(X_train, augment=True)
    return datagen

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

train_test_split = int(1 * len(train_ids))

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                               preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
    
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

data_generator = create_data_generator(X_train, Y_train)

'''
X_augmented = []
Y_augmented = []
for i in range(train_test_split):
    aug_iter = data_generator.flow(np.expand_dims(X_train[i], 0), batch_size=1, seed=seed+i)
    for _ in range(AUGMENTATIONS_PER_IMG):
        X_augmented.append(np.squeeze(next(aug_iter)[0].astype(np.uint8)))
        
    aug_iter = data_generator.flow(np.expand_dims(Y_train[i], 0), batch_size=1, seed=seed+i)
    for _ in range(AUGMENTATIONS_PER_IMG):
        Y_augmented.append(next(aug_iter)[0].astype(np.bool))
        
X_augmented = np.array(X_augmented)
Y_augmented = np.array(Y_augmented)
shuffled_indices = np.arange(X_augmented.shape[0])
np.random.shuffle(shuffled_indices)
X_augmented = X_augmented[shuffled_indices]
Y_augmented = Y_augmented[shuffled_indices]

X_val = X_train[train_test_split:]
Y_val = Y_train[train_test_split:]
'''

ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1,1), activation='sigmoid') (c9)

'''
def cell_classifier():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)
    
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    
    f1 = Dense(1024, activation='tanh') (c5)
    f1 = Dropout(0.4) (f1)
    
    outputs = Dense(, activation='softmax')(f1)
    return inputs, outputs

def create_cell_classifier_model():
    
'''


def create_model():
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    return model

def get_model(path='model-dsbowl2018-1.h5'):
    return load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

def train(model, nb_epochs=50):
    # Fit model
    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
    normal_generator = ImageDataGenerator()
    #results = model.fit(X_augmented, Y_augmented, validation_data=(X_val, Y_val), batch_size=32, epochs=nb_epochs,
    #                callbacks=[earlystopper, checkpointer])
    X_train_augmented = data_generator.flow(X_train, batch_size=32, shuffle=True, seed=seed)
    Y_train_augmented = data_generator.flow(Y_train, batch_size=32, shuffle=True, seed=seed)
    X_val = normal_generator.flow(X_train, batch_size=32, shuffle=True, seed=seed)
    Y_val = normal_generator.flow(Y_train, batch_size=32, shuffle=True, seed=seed)
    train_generator = zip(X_train_augmented, Y_train_augmented)
    val_generator = zip(X_val, Y_val)
    results = model.fit_generator(train_generator,
                        steps_per_epoch=train_test_split / 32, epochs=nb_epochs,
                        callbacks=[earlystopper, checkpointer], validation_data=val_generator,
                        validation_steps=int(len(train_ids) / 32))
    
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


from skimage.measure import regionprops
import cv2
def split_mask(mask):
    thresh = mask.copy().astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    i = 0
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            hull = cv2.convexHull(contour, returnPoints = False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            points = []
            dd = []
            
            # Gather all defect points
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                d = d / 256
                dd.append(d)
                
            for i in range(len(dd)):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                if dd[i] > 1.0 and dd[i]/np.max(dd) > 0.2:
                    points.append(f)
                    
            i = i + 1
            if len(points) >= 2:
                for i in range(len(points)):
                    f1 = points[i]
                    p1 = tuple(contour[f1][0])
                    nearest = None
                    min_dist = np.inf
                    for j in range(len(points)):
                        if i != j:
                            f2 = points[j]
                            p2 = tuple(contour[f2][0])
                            dist = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) 
                            if dist < min_dist:
                                min_dist = dist
                                nearest = p2
                                
                    cv2.line(thresh, p1, nearest, [0, 0, 0], 2)
    return thresh

def split_and_relabel(mask):
    masks = []
    for i in np.unique(mask):
        if i == 0: # id = 0 is for background
            continue
        mask_i = (mask==i).astype(np.uint8)
        props = regionprops(mask_i, cache=False)
        if len(props) > 0:
            prop = props[0]
            if prop.convex_area/prop.filled_area > 1.1:
                mask_i = split_mask(mask_i)
        masks.append(mask_i)
        
    masks = np.array(masks)
    masks_combined = np.amax(masks, axis=0)
    labels = label(masks_combined)
    return labels
  
'''
def separate_nucei(lab_img):
    ind_masks = []
    for i in unique(labeled_mask):
        if i == 0:
            continue
        ind_masks.append((labeled_mask==i).astype(np.uint8))
    for mask in ind_masks:
'''    
   

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    #x = split_and_relabel(x)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
def create_submission_file(preds):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('submission.csv')    
        
def show_results(results):
    for result in results:
        imshow(result)
    
def predict():
    # Predict on train, val, and test
    model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)
    
    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))
    
    '''    
    for idx in range(len(X_test)):
        imshow(X_test[idx])
        plt.show()
        imshow(np.squeeze(preds_test_t[idx]))
        plt.show()
    '''   
   
    return preds_train_t, preds_val_t, preds_test_upsampled

#model = get_model()
#model = create_model()
#train(model, 1000)
#results = predict()[2]
#show_results(results)
#create_submission_file(results)
