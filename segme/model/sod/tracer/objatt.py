import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.sequence import Sequenсe


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class ObjectAttention(layers.Layer):
    def __init__(self, denoise, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # encoder map
            layers.InputSpec(ndim=4, axes={-1: 1})  # decoder map
        ]

        self.denoise = denoise

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if not self.channels // 8:
            raise ValueError('Channel dimension should be greater then 8.')

        self.conv_in = Sequenсe([
            ConvNormAct(None, 3),
            ConvNormAct(self.channels // 2, 1)
        ])
        self.conv_mid0 = Sequenсe([
            ConvNormAct(None, 1),
            ConvNormAct(self.channels // 8, 1)
        ])
        self.conv_mid1 = Sequenсe([
            ConvNormAct(None, 3),
            ConvNormAct(self.channels // 8, 1)
        ])
        self.conv_mid2 = Sequenсe([
            ConvNormAct(None, 3, dilation_rate=3),
            ConvNormAct(self.channels // 8, 1)
        ])
        self.conv_mid3 = Sequenсe([
            ConvNormAct(None, 3, dilation_rate=5),
            ConvNormAct(self.channels // 8, 1)
        ])
        self.conv_out = ConvNormAct(1, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        encoder_map, decoder_map = inputs

        foregrounds = tf.nn.sigmoid(decoder_map)
        backgrounds = 1. - foregrounds
        edges = backgrounds * tf.cast(backgrounds < self.denoise, self.compute_dtype)

        outputs = (foregrounds + edges) * encoder_map

        outputs = self.conv_in(outputs)
        outputs += tf.concat([
            self.conv_mid0(outputs), self.conv_mid1(outputs), self.conv_mid2(outputs), self.conv_mid3(outputs)],
            axis=-1)
        outputs = self.conv_out(outputs)

        outputs = tf.nn.relu(outputs)
        outputs += decoder_map

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super().get_config()
        config.update({'denoise': self.denoise})

        return config
