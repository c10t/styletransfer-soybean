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
