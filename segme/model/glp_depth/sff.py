import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu, SameConv


@register_keras_serializable(package='SegMe>MINet')
class SelectiveFeatureFusion(layers.Layer):
    def __init__(self, standardized=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # local
            layers.InputSpec(ndim=4)  # global
        ]

        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if self.channels != input_shape[1][-1]:
            raise ValueError('Channel dimension of the inputs should be equal.')

        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels}),
            layers.InputSpec(ndim=4, axes={-1: self.channels})
        ]

        self.conv0 = ConvNormRelu(self.channels, 3, standardized=self.standardized)
        self.conv1 = ConvNormRelu(self.channels // 2, 3, standardized=self.standardized)
        self.conv2 = SameConv(2, 3, activation='sigmoid')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        attention = tf.concat(inputs, axis=-1)
        attention = self.conv0(attention)
        attention = self.conv1(attention)
        attention = self.conv2(attention)
        attention = tf.split(attention, 2, axis=-1)

        outputs = inputs[0] * attention[0] + inputs[1] * attention[1]

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({'standardized': self.standardized})

        return config
