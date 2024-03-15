import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Model>Matting>FBAMatting')
class Twomap(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap

    def call(self, inputs, **kwargs):
        twomap = twomap_transform(inputs)
        twomap = tf.cast(twomap, self.compute_dtype)

        return twomap

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (2,)


def twomap_transform(trimap):
    twomap = tf.concat([trimap == 0, trimap == 255], axis=-1)
    twomap = tf.cast(twomap, 'uint8') * 255

    return twomap
