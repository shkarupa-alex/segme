import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.image import euclidean_dist_transform


@register_keras_serializable(package='SegMe>FBAMatting')
class Distance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap

    def call(self, inputs, **kwargs):
        outputs = distance_transform(inputs)
        outputs = tf.cast(outputs, self.compute_dtype)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (6,)


def distance_transform(trimap, length=320):
    clicks = []
    for value in [0, 255]:
        twomap = tf.cast(trimap != value, 'uint8') * 255
        distance = -euclidean_dist_transform(twomap, dtype='float32') ** 2
        clicks.extend([
            tf.exp(distance / (2 * (0.02 * length) ** 2)),
            tf.exp(distance / (2 * (0.08 * length) ** 2)),
            tf.exp(distance / (2 * (0.16 * length) ** 2)),
        ])

    clicks = tf.concat(clicks, axis=-1)
    clicks = tf.cast(tf.round(clicks * 255.), dtype='uint8')

    return clicks
