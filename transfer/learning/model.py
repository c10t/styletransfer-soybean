from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization

def residual_block(input_ts):
    x = Conv2D(128, (3, 3), strides=1, padding='same')(input_ts)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, input_ts])
