#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from pathlib import Path
from pillow import Image
from keras.models import load_model

MODEL_PATH = "logdir/model_file.hdf5"
IMAGES_FOLDER = "sample_images"
IMAGE_SHAPE = (32, 32, 3)


def crop_resize(image_path):
    image = Image.open(image_path)
    length = min(image.size)
    crop = image.crop((0, 0, length, length))
    resized = crop.resize(IMAGE_SHAPE[:2])  # use width * height
    img = np.array(resized).astype("float32")
    img /= 255
    return img


if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    folder = Path(IMAGES_FOLDER)
    image_paths = [str(f) for f in folder.glob("*.png")]
    images = [crop_resize(p) for p in image_paths]
    images = np.asarray(images)

    predicted = model.predict_classes(images)

    assert predicted[0] == 3, "image should be cat"
    assert predicted[1] == 5, "image should be dog"

    print("You can detect cat & dog!")
