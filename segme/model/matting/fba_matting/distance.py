import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from tfmiss.image import euclidean_distance


@register_keras_serializable(package="SegMe>Model>Matting>FBAMatting")
class Distance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(
            ndim=4, axes={-1: 1}, dtype="uint8"
        )  # trimap

    def call(self, inputs, **kwargs):
        outputs = distance_transform(inputs)
        outputs = tf.cast(outputs, self.compute_dtype)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (6,)


def distance_transform(trimap, length=320):
    clicks = []
    for value in [0, 255]:
        twomap = tf.cast(trimap != value, "uint8") * 255
        distance = -euclidean_distance(twomap) ** 2
        clicks.extend(
            [
                tf.exp(distance / (2 * (0.02 * length) ** 2)),
                tf.exp(distance / (2 * (0.08 * length) ** 2)),
                tf.exp(distance / (2 * (0.16 * length) ** 2)),
            ]
        )

    clicks = tf.concat(clicks, axis=-1)
    clicks = tf.saturate_cast(tf.round(clicks * 255.0), dtype="uint8")

    return clicks
