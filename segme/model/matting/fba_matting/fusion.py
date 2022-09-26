import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Model>Matting>FBAMatting')
class Fusion(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image
            layers.InputSpec(ndim=4, axes={-1: 7}),  # decoder output
        ]
        self.la = 0.1

    def call(self, inputs, **kwargs):
        image, alfgbg = inputs
        alpha, foreground, background = tf.split(alfgbg, [1, 3, 3], axis=-1)

        alpha_sqr = alpha ** 2
        foreground = alpha * (image - background) + alpha_sqr * (background - foreground) + foreground
        background = alpha * (2 * background - image - foreground) - alpha_sqr * (background - foreground) + image
        foreground = tf.clip_by_value(foreground, 0., 1.)
        background = tf.clip_by_value(background, 0., 1.)

        imbg_diff = image - background
        fgbg_diff = foreground - background
        alpha_numer = alpha * self.la + tf.reduce_sum(imbg_diff * fgbg_diff, axis=-1, keepdims=True)
        alpha_denom = tf.reduce_sum(fgbg_diff ** 2, axis=-1, keepdims=True) + self.la
        alpha = tf.clip_by_value(alpha_numer / alpha_denom, 0., 1.)

        return alpha, foreground, background

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        base_shape = input_shape[0][:-1]
        return base_shape + (1,), base_shape + (3,), base_shape + (3,)
