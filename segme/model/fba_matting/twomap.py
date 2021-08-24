import tensorflow as tf
from keras import layers, utils
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.image import euclidean_dist_transform


@register_keras_serializable(package='SegMe>FBAMatting')
class Twomap(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap

    def call(self, inputs, **kwargs):
        twomap = layers.concatenate([
            tf.cast(inputs == 0, self.compute_dtype),
            tf.cast(inputs == 255, self.compute_dtype)
        ], axis=-1)

        return twomap

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (2,)
