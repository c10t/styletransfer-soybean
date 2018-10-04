#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os

import numpy as np
# import matplotlib.pyplot as plt

# from urllib.request import urlretrieve

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K


class TransferDefinition():
    def __init__(self, content_image_path, style_image_path, img_nrows=400):
        self.width, self.height = load_img(content_image_path).size
        self.img_nrows = img_nrows
        self.img_ncols = int(self.width * self.img_nrows / self.height)

    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=(self.img_nrows, self.img_ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def deprocess_image(self, x):
        img = x.copy()
        img = img.reshape(self.img_nrows, self.img_ncols, 3)

        # Remove zero-center by mean pixel
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68

        # BGR -> RGB
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype("uint8")

        return img


def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def content_loss(content, combination):
    return K.sum(K.square(combination - content))


def style_loss(tdef, style, combination):
    return K.sum()


def total_variation_loss(tdef, x):
    assert K.ndim(x) == 4
    r = tdef.img_nrows - 1
    c = tdef.img_ncols - 1
    a = K.square(x[:, :r, :c, :] - x[:, 1:, :c, :])
    b = K.square(x[:, :r, :c, :] - x[:, :r, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


if __name__ == '__main__':
    print("not implemented")