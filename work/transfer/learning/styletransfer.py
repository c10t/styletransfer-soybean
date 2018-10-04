#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

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
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = tdef.img_nrows * tdef.img_ncols
    return K.sum(K.square(S - C) / (4. * (channels ** 2) * (size ** 2)))


def total_variation_loss(tdef, x):
    assert K.ndim(x) == 4
    r = tdef.img_nrows - 1
    c = tdef.img_ncols - 1
    a = K.square(x[:, :r, :c, :] - x[:, 1:, :c, :])
    b = K.square(x[:, :r, :c, :] - x[:, :r, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def main(content_image_path, style_image_path, iter_count=10,
         content_weight=1.0, style_weight=0.11,
         total_valiation_weight=0.001, learning_rate=0.001):
    tdef = TransferDefinition(content_image_path, style_image_path)

    # inputs
    content_image = K.variable(tdef.preprocess_image(content_image_path))
    style_image = K.variable(tdef.preprocess_image(style_image_path))

    # generated image
    combination_image = K.placeholder(
      (1, tdef.img_nrows, tdef.img_ncols, 3)
    )

    input_tensor = K.concatenate(
      [content_image, style_image, combination_image], axis=0
    )

    # load pre-trained model
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights="imagenet", include_top=False)

    outputs_dict = dict(
      [(layer.name, layer.output) for layer in model.layers]
    )

    # define loss
    loss = K.variable(0.)
    feature_map = outputs_dict["block5_conv2"]
    feature_of_content = feature_map[0, :, :, :]
    feature_of_combination = feature_map[2, :, :, :]

    loss += content_weight * content_loss(
      feature_of_content, feature_of_combination
    )

    feature_layers = ["block{}_conv1".format(i) for i in range(1, 6)]

    for layer_name in feature_layers:
        feature_map = outputs_dict[layer_name]
        feature_of_style = feature_map[1, :, :, :]
        feature_of_combination = feature_map[2, :, :, :]
        sl = style_loss(tdef, feature_of_style, feature_of_combination)
        loss += (style_weight / len(feature_layers)) * sl

    loss += total_valiation_weight * total_variation_loss(
      tdef, combination_image
    )

    grads = K.gradients(loss, combination_image)[0]
    style_transfer = K.function([combination_image], [loss, grads])

    # optimize loss
    image = tdef.preprocess_image(content_image_path)
    for i in range(iter_count):
        print("Start of iteration {}".format(i + 1))
        loss_value, grad_values = style_transfer([image])
        image -= grad_values * learning_rate

    plt.figure(figsize=(10, 5))
    for kind in ["original", "style", "styled"]:
        if kind == "original":
            img = load_img(
              content_image_path,
              target_size=(tdef.img_nrows, tdef.img_ncols)
            )
            ax = plt.subplot(1, 3, 1)

        elif kind == "style":
            img = load_img(
              style_image_path,
              target_size=(tdef.img_nrows, tdef.img_ncols)
            )
            ax = plt.subprot(1, 3, 2)

        elif kind == "styled":
            img = tdef.deprocess_image(image)
            ax = plt.subplot(1, 3, 3)

        ax.set_title(kind)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    dir_name, file_name = os.path.split(content_image_path)
    file_root, ext = os.path.splitext(file_name)
    plt.savefig(os.path.join(dir_name, file_root + "_styled.png"))
    plt.show()


if __name__ == '__main__':
    # TODO: use argparse
    image_url = "X"
    content_path = os.path.join(
      os.path.dirname(__file__),
      "data/content.jpg"
    )
    urlretrieve(image_url, content_path)

    image_url = "Y"
    style_path = os.path.join(
      os.path.dirname(__file__),
      "data/style.jpg"
    )
    urlretrieve(image_url, style_path)

    main(content_path, style_path)
