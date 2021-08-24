import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe')
class ToChannelLast(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return tf.transpose(inputs, [0, 2, 3, 1])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:] + (input_shape[1],)


@register_keras_serializable(package='SegMe')
class ToChannelFirst(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        return tf.transpose(inputs, [0, 3, 1, 2])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3]) + input_shape[1:3]


def to_channel_last(inputs, **kwargs):
    return ToChannelLast(**kwargs)(inputs)


def to_channel_first(inputs, **kwargs):
    return ToChannelFirst(**kwargs)(inputs)
