from keras.src import layers
from keras.src import ops
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

    def call(self, inputs, training=False, **kwargs):
        image, alfgbg = inputs
        alfgbg = ops.cast(alfgbg, "float32")
        alpha, foreground, background = ops.split(alfgbg, [1, 4], axis=-1)

        if training:
            return alfgbg, alpha, foreground, background

        image = ops.cast(image, "float32")

        alpha = ops.clip(alpha, 0.0, 1.0)
        foreground = ops.sigmoid(foreground)
        background = ops.sigmoid(background)

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
        foreground = ops.clip(foreground, 0.0, 1.0)
        background = ops.clip(background, 0.0, 1.0)

        imbg_diff = image - background
        fgbg_diff = foreground - background
        alpha_numer = alpha * self.la + ops.sum(
            imbg_diff * fgbg_diff, axis=-1, keepdims=True
        )
        alpha_denom = ops.sum(fgbg_diff**2, axis=-1, keepdims=True) + self.la
        alpha = ops.clip(alpha_numer / alpha_denom, 0.0, 1.0)

        alfgbg = ops.concatenate([alpha, foreground, background], axis=-1)

        return alfgbg, alpha, foreground, background

    def compute_output_shape(self, input_shape):
        base_shape = input_shape[0][:-1]
        return (
            base_shape + (7,),
            base_shape + (1,),
            base_shape + (3,),
            base_shape + (3,),
        )
