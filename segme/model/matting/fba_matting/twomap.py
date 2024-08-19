import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Model>Matting>FBAMatting")
class Twomap(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(
            ndim=4, axes={-1: 1}, dtype="uint8"
        )  # trimap

    def call(self, inputs, **kwargs):
        twomap = twomap_transform(inputs)
        twomap = tf.cast(twomap, self.compute_dtype)

        return twomap

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (2,)


def twomap_transform(trimap):
    twomap = tf.concat([trimap == 0, trimap == 255], axis=-1)
    twomap = tf.cast(twomap, "uint8") * 255

    return twomap
