import os

from time import gmtime, strftime
from keras.callbacks import TensorBoard


def make_tensorboard(set_dir_name=''):
    tictoc = strftime("%Y%m%d%H%M%S", gmtime())
    directory_name = tictoc
    log_dir = 'tb_' + set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    return tensorboard
