import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import AtrousSeparableConv, ConvNormRelu


@register_keras_serializable(package='SegMe>Tracer')
class ObjectAttention(layers.Layer):
    def __init__(self, kernel_size, denoise, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # encoder map
            layers.InputSpec(ndim=4, axes={-1: 1})  # decoder map
        ]

        self.kernel_size = kernel_size
        self.denoise = denoise

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if not self.channels // 8:
            raise ValueError('Channel dimension should be greater then 8.')

        self.conv_in = AtrousSeparableConv(self.channels // 2, self.kernel_size, activation='selu')
        self.conv_mid0 = AtrousSeparableConv(self.channels // 8, 1, activation='selu')
        self.conv_mid1 = AtrousSeparableConv(self.channels // 8, 3, activation='selu')
        self.conv_mid2 = AtrousSeparableConv(self.channels // 8, 3, activation='selu', dilation_rate=3)
        self.conv_mid3 = AtrousSeparableConv(self.channels // 8, 3, activation='selu', dilation_rate=5)
        self.conv_out = ConvNormRelu(1, 1, activation='selu')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        encoder_map, decoder_map = inputs

        foregrounds = tf.nn.sigmoid(decoder_map)
        backgrounds = 1. - foregrounds
        edges = tf.where(backgrounds < self.denoise, backgrounds, 0.)

        outputs = (foregrounds + edges) * encoder_map

        outputs = self.conv_in(outputs)
        outputs += tf.concat([
            self.conv_mid0(outputs), self.conv_mid1(outputs), self.conv_mid2(outputs), self.conv_mid3(outputs)],
            axis=-1)
        outputs = self.conv_out(outputs)

        outputs = tf.nn.relu(outputs) + decoder_map

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'denoise': self.denoise
        })

        return config
