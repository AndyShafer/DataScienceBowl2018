# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 00:18:35 2018

@author: Andy
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from skimage.morphology import label
from skimage.measure import regionprops
import keras.backend
import keras.preprocessing.image
import numpy
import skimage.io
import skimage.transform

import keras
from keras.layers import Input
import keras_rcnn
import keras_rcnn.models

dsb_data_dir = './input_stage1/'
stage_label = 'stage1'

train_labels = pd.read_csv(os.path.join(dsb_data_dir,'{}_train_labels.csv'.format(stage_label)))
train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])
train_labels.sample(3)

TRAIN_PATH = './input/train/'
TEST_PATH = './input/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*'))

#print(all_images)
img_df = pd.DataFrame({'path': all_images})
img_id = lambda in_path: in_path.split('\\')[-3]
img_type = lambda in_path: in_path.split('\\')[-2]
img_group = lambda in_path: in_path.split('\\')[-4].split('_')[1]
img_stage = lambda in_path: in_path.split('\\')[-4].split('_')[0]
#img_stage = lambda in_path: 'stage1'
img_df['ImageId'] = img_df['path'].map(img_id)
img_df['ImageType'] = img_df['path'].map(img_type)
img_df['TrainingSplit'] = img_df['path'].map(img_group)
img_df['Stage'] = img_df['path'].map(img_stage)
img_df.sample(2)

train_df = img_df.query('TrainingSplit=="train"')
train_rows = []
group_cols = ['ImageId']
for n_group, n_rows in train_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['mask_paths'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
    c_row['image_paths'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    
    train_rows += [c_row]
train_img_df = pd.DataFrame(train_rows)
IMG_CHANNELS = 3
def read_and_stack(in_img_list):
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0) / 255.0
train_img_df['images'] = train_img_df['image_paths'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])
def read_and_stack_masks(in_img_list):
    return np.sum(np.stack([i*(imread(c_img)>0) for i, c_img in 
                            enumerate(in_img_list,1)], 0), 0)
print(len(train_img_df['mask_paths']))
print(train_img_df['mask_paths'])
train_img_df['masks'] = train_img_df['mask_paths'].map(read_and_stack_masks).map(lambda x: x.astype(int))
train_img_df.sample(1)

n_img = 6
fig, m_axs = plt.subplots(2, n_img, figsize = (12, 4))
for(_, c_row), (c_im, c_lab) in zip(train_img_df.sample(n_img).iterrows(),  m_axs.T):
    c_im.imshow(c_row['images'])
    c_im.axis('off')
    c_im.set_title('Microscope')
    
    c_lab.imshow(c_row['masks'])
    c_lab.axis('off')
    c_lab.set_title('Labeled')

train_img_df['images'].map(lambda x: x.shape).value_counts()
train_img_df.sample(1)

train_dict = [dict(image = dict(
        pathname = c_row['image_paths'][0],
            shape = dict(zip(['r', 'c', 'channels'], c_row['images'].shape))),
            objects = [{'bounding_box' : dict(
                    minimum = dict(r = c_reg.bbox[0], c = c_reg.bbox[1]),
                    maximum = dict(r = c_reg.bbox[2], c = c_reg.bbox[3])),
                        'class' : "nucleus",
                        'mask' : dict(pathname = c_path)
                    }
                        for c_reg, c_path in zip(
                            regionprops(label(c_row['masks'])),
                            c_row['mask_paths']
                        )]
                    )
                for _, c_row in train_img_df.iterrows()]
print(str(train_dict[0])[:500])


def scale_size(size, min_size, max_size):
    """
    Rescales a given image size such that the larger axis is
    no larger than max_size and the smallest axis is as close
    as possible to min_size.
    """
    assert (len(size) == 2)

    scale = min_size / numpy.min(size)

    # Prevent the biggest axis from being larger than max_size.
    if numpy.round(scale * numpy.max(size)) > max_size:
        scale = max_size / numpy.max(size)

    rows, cols = size
    rows *= scale
    cols *= scale

    return (int(rows), int(cols)), scale


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(
            self,
            dictionary,
            classes,
            generator,
            target_shape=None,
            scale=1,
            ox=None,
            oy=None,
            batch_size=1,
            shuffle=False,
            seed=None
    ):
        self.dictionary = dictionary
        self.classes = classes
        self.generator = generator

        r = dictionary[0]["image"]["shape"]["r"]
        c = dictionary[0]["image"]["shape"]["c"]

        channels = dictionary[0]["image"]["shape"]["channels"]

        self.image_shape = (r, c, channels)
        self.scale = scale
        self.ox = ox
        self.oy = oy

        self.batch_size = batch_size

        if target_shape is None:
            self.target_shape, self.scale = scale_size(self.image_shape[0:2], numpy.min(self.image_shape[:2]), numpy.max(self.image_shape[:2]))

            self.target_shape = self.target_shape + (self.image_shape[2],)

        else:
            self.target_shape = target_shape + (self.image_shape[2],)

        # Metadata needs to be computed only once.
        r, c, channels = self.target_shape

        self.target_metadata = numpy.array([[r, c, self.scale]])

        super(DictionaryIterator, self).__init__(len(self.dictionary), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            selection = next(self.index_generator)

        return self._get_batches_of_transformed_samples(selection)

    def _get_batches_of_transformed_samples(self, selection):
        # Labels has num_classes + 1 elements, since 0 is reserved for
        # background.
        num_classes = len(self.classes)

        target_bounding_boxes = numpy.zeros((self.batch_size, 0, 4), dtype=keras.backend.floatx())

        target_images = numpy.zeros((self.batch_size,) + self.target_shape, dtype=keras.backend.floatx())

        target_masks = numpy.zeros((self.batch_size,) + self.target_shape, dtype=keras.backend.floatx())

        target_scores = numpy.zeros((self.batch_size, 0, num_classes + 1), dtype=numpy.uint8)

        for batch_index, image_index in enumerate(selection):
            count = 0

            while count == 0:
                # Image
                target_image_pathname = self.dictionary[image_index]["image"]["pathname"]

                target_image = skimage.io.imread(target_image_pathname)[:, :, :3]

                if target_image.ndim == 2:
                    target_image = numpy.expand_dims(target_image, -1)

                # crop
                if self.ox is None:
                    offset_x = numpy.random.randint(0, self.image_shape[1] - self.target_shape[1] + 1)
                else:
                    offset_x = self.ox

                if self.oy is None:
                    offset_y = numpy.random.randint(0, self.image_shape[0] - self.target_shape[0] + 1)
                else:
                    offset_y = self.oy

                target_image = target_image[offset_y:self.target_shape[0] + offset_y, offset_x:self.target_shape[1] + offset_x, :]

                # Copy image to batch blob.
                target_images[batch_index] = skimage.transform.rescale(target_image, scale=self.scale, mode="reflect")

                # Set ground truth boxes.
                for object_index, b in enumerate(self.dictionary[image_index]["objects"]):
                    if b["class"] not in self.classes:
                        continue

                    bounding_box = b["bounding_box"]

                    minimum_c = bounding_box["minimum"]["c"] - offset_x
                    minimum_r = bounding_box["minimum"]["r"] - offset_y

                    maximum_c = bounding_box["maximum"]["c"] - offset_x
                    maximum_r = bounding_box["maximum"]["r"] - offset_y

                    if maximum_c == target_image.shape[1]:
                        maximum_c -= 1

                    if maximum_r == target_image.shape[0]:
                        maximum_r -= 1

                    if minimum_c >= 0 and maximum_c < target_image.shape[1] and minimum_r >= 0 and maximum_r < target_image.shape[0]:
                        count += 1

                        target_bounding_box = [
                            minimum_c,
                            minimum_r,
                            maximum_c,
                            maximum_r
                        ]

                        target_bounding_boxes = numpy.append(target_bounding_boxes, [[target_bounding_box]], axis=1)

                        target_score = [0] * (num_classes + 1)

                        target_score[self.classes[b["class"]]] = 1

                        target_scores = numpy.append(target_scores, [[target_score]], axis=1)

            # Scale the ground truth boxes to the selected image scale.
            target_bounding_boxes[batch_index, :, :4] *= self.scale

        return [target_bounding_boxes, target_images, target_masks, self.target_metadata, target_scores], None


class ImageSegmentationGenerator:
    def flow(self, dictionary, classes, target_shape=None, scale=None, ox=None, oy=None, batch_size=1, shuffle=True, seed=None):
        return DictionaryIterator(dictionary, classes, self, target_shape, scale, ox, oy, batch_size, shuffle, seed)
    

seg_gen = ImageSegmentationGenerator().flow(train_dict, {'nucleus': 1})
x_out, _ = seg_gen.next()
print('x', len(x_out))
for x in x_out:
    print('\t', x.shape)

(target_bounding_boxes, target_image, _, _, target_scores) = x_out
target_bounding_boxes = np.squeeze(target_bounding_boxes)
target_image = np.squeeze(target_image)
target_scores = np.argmax(target_scores, -1)
target_scores = np.squeeze(target_scores)

_, axis = plt.subplots(1, figsize=(12, 8))
axis.imshow(target_image)

for target_index, target_score in enumerate(target_scores):
    if target_score > 0:
        xy = [
            target_bounding_boxes[target_index][0],
            target_bounding_boxes[target_index][1]
        ]

        w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
        h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]

        rectangle = matplotlib.patches.Rectangle(xy, w, h, 
                                                 edgecolor=plt.cm.RdBu(target_score), 
                                                 facecolor="none")
        axis.add_patch(rectangle)

matplotlib.pyplot.show()

img_in = Input((None, None, 3))
rcnn_model = keras_rcnn.models.RCNN(img_in, 2)
optimizer = keras.optimizers.Adam(0.0001)
rcnn_model.compile(optimizer)
print('Input:', rcnn_model.get_input_shape_at(0))
print('Output:', rcnn_model.get_output_shape_at(0))

def fit_gen():
    for(boxes, target_img, img_b, meta_out, target_scores), _ in seg_gen:
        
        yield [boxes, target_img, target_scores, meta_out], None


rcnn_model.fit_generator(fit_gen(), steps_per_epoch = 20, epochs = 5)

test_ds = next(fit_gen())
y_anchors, y_scores = rcnn_model.predict(test_ds[0])
print(y_anchors.shape, y_scores.shape)

target_bounding_boxes = np.squeeze(y_anchors)
target_image = np.squeeze(test_ds[0][1][0])
target_scores = np.argmax(np.squeeze(y_scores), -1)

_, axis = plt.subplots(1, figsize=(12, 8))

axis.imshow(target_image)

for target_index, target_score in enumerate(target_scores):
    if target_score > 0:
        xy = [
            target_bounding_boxes[target_index][0],
            target_bounding_boxes[target_index][1]
        ]

        w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
        h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]

        rectangle = matplotlib.patches.Rectangle(xy, w, h, 
                                                 edgecolor=plt.cm.RdBu(target_score), 
                                                 facecolor="none")
        axis.add_patch(rectangle)

matplotlib.pyplot.show()
