import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class GlobalAverage(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    @shape_type_conversion
    def build(self, input_shape):
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.pool(inputs)
        outputs = tf.broadcast_to(outputs, tf.shape(inputs))

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
