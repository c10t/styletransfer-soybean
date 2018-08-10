# import os
# import glob
# import math
# import random

import numpy as np

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model


def residual_block(input_tensor):
    x = Conv2D(128, (3, 3), strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, input_tensor])


def build_encoder_decoder(input_shape=(224, 224, 3)):
    # Encoder part
    input_tensor = Input(shape=input_shape, name="input")

    # Normalize to range [0, 1]
    x = Lambda(lambda a: a / 255.)(input_tensor)

    x = Conv2D(32, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # add 5 residual blocks
    for _ in range(5):
        x = residual_block(x)
    
    # Decoder part
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3, (9, 9), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # scake change to range [0, 255]
    output = Lambda(lambda a: (a + 1) * 127.5)(x)

    return Model(inputs=[input_tensor], outputs=[output])

tmp_input_shape = (224, 224, 3)

generated_model = build_encoder_decoder(input_shape=tmp_input_shape)

# loss network

from tensorflow.keras.applications.vgg16 import VGG16

vgg16 = VGG16()


# change settings no to train the model
for layer in vgg16.layers:
    layer.trainable = False


def normalizer_vgg16(x):
    """
    do RGB -> BGR and asymptotically centralize
    """
    return (x[:, :, :, ::-1] - 120) / 255.


# define the name of layers from which we try to extract the features 
style_layer_names = (
    'block1_conv2',
    'block2_conv2',
    'block3_conv3',
    'block4_conv4'
)

contents_layer_names = (
    'block3_conv3'
)

# list for keep the inner layer's output
outputs_inner_style = []
outputs_inner_contents = []

generated_inputs = generated_model.output

z = Lambda(normalizer_vgg16)(generated_inputs)

for layer in vgg16.layers:
    z = layer(z)
    if layer.name in style_layer_names:
        outputs_inner_style.append(z)
    if layer.name in contents_layer_names:
        outputs_inner_contents.append(z)

model = Model(inputs=generated_model.input,
    outputs=outputs_inner_style+outputs_inner_contents)

# prepare answer
tmp_input_size = tmp_input_shape[:2]

style_image = load_img('img/style/sample.png', target_size=input_size)

array_style_image = np.expand_dims(img_to_array(style_image), axis=0)

# define input layer
input_style = Input(shape=input_shape, name='input_style')

style_outputs = []
x = Lambda(normalizer_vgg16)(input_style)

for layer in vgg16.layers:
    x = layer(x)
    if layer.name in style_layer_names:
        style_outputs.append(x)

style_model = Model(inputs=input_style, outputs=style_outputs)

y_true_style = style_model.predict(array_style_image)
