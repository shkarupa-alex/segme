import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Model>Matting>FBAMatting")
class Fusion(layers.Layer):
    def __init__(self, dtype="float32", **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: 3}),  # image
            InputSpec(ndim=4, axes={-1: 7}),  # alpha-fg-bg
        ]
        self.la = 0.1

    def call(self, inputs, **kwargs):
        image, alfgbg = inputs
        image = tf.cast(image, "float32")
        alfgbg = tf.cast(alfgbg, "float32")

        alpha, foreground, background = tf.split(alfgbg, [1, 3, 3], axis=-1)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        foreground = tf.nn.sigmoid(foreground)
        background = tf.nn.sigmoid(background)

        # TODO: https://github.com/MarcoForte/FBA_Matting/issues/55
        alpha_sqr = alpha**2
        foreground = (
            alpha * (image - background)
            + alpha_sqr * (background - foreground)
            + foreground
        )
        background = (
            alpha * (2 * background - image - foreground)
            - alpha_sqr * (background - foreground)
            + image
        )
        foreground = tf.clip_by_value(foreground, 0.0, 1.0)
        background = tf.clip_by_value(background, 0.0, 1.0)

        imbg_diff = image - background
        fgbg_diff = foreground - background
        alpha_numer = alpha * self.la + tf.reduce_sum(
            imbg_diff * fgbg_diff, axis=-1, keepdims=True
        )
        alpha_denom = (
            tf.reduce_sum(fgbg_diff**2, axis=-1, keepdims=True) + self.la
        )
        alpha = tf.clip_by_value(alpha_numer / alpha_denom, 0.0, 1.0)

        alfgbg = tf.concat([alpha, foreground, background], axis=-1)

        return alfgbg, alpha, foreground, background

    def compute_output_shape(self, input_shape):
        base_shape = input_shape[0][:-1]
        return (
            base_shape + (7,),
            base_shape + (1,),
            base_shape + (3,),
            base_shape + (3,),
        )
