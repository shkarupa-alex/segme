import tensorflow as tf
from keras.applications import imagenet_utils
from keras.mixed_precision import global_policy


def preprocess_input(inputs, mode='torch', name=None):
    with tf.name_scope(name or 'preprocess_input'):
        outputs = tf.cast(inputs, global_policy().compute_dtype)
        outputs = imagenet_utils.preprocess_input(outputs, data_format='channels_last', mode=mode)

    return outputs
