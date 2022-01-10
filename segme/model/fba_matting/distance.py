import tensorflow as tf
from keras import layers, utils
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.image import euclidean_dist_transform


@register_keras_serializable(package='SegMe>FBAMatting')
class Distance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: 2})  # twomap
        self.length = 320

    def call(self, inputs, **kwargs):
        clicks = []
        for channel in range(2):
            restored = tf.cast((1. - inputs[..., channel:channel + 1]) * 255., 'uint8')
            distance = -euclidean_dist_transform(restored, dtype='float32') ** 2
            clicks.extend([
                tf.exp(distance / (2 * (0.02 * self.length) ** 2)),
                tf.exp(distance / (2 * (0.08 * self.length) ** 2)),
                tf.exp(distance / (2 * (0.16 * self.length) ** 2)),
            ])

        clicks = tf.concat(clicks, axis=-1)
        clicks = tf.cast(clicks, dtype=self.compute_dtype)
        clicks = tf.stop_gradient(clicks)

        return clicks

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (6,)
