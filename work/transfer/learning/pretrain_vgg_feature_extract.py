#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras.preprocessing.image as Image

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


base_model = VGG19(weights="imagenet")
model = Model(
  inputs=base_model.input,
  outputs=base_model.get_layer("block4_pool").output
)

img_path = "elephant.jpg"
img = Image.load_img(img_path, target_size=(224, 224))

x = Image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
