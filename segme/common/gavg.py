import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.shape import get_shape


@register_keras_serializable(package="SegMe>Common")
class GlobalAverage(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        self.pool = layers.GlobalAveragePooling2D(
            keepdims=True, dtype=self.dtype_policy
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.pool(inputs)

        shape, _ = get_shape(inputs)
        outputs = tf.broadcast_to(outputs, shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
