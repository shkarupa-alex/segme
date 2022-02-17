import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .uniatt import UnionAttention
from ...common import ConvNormRelu, ResizeByScale


@register_keras_serializable(package='SegMe>Tracer')
class Aggregation(layers.Layer):
    def __init__(self, confidence, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(3)]

        self.confidence = confidence

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.upsample = ResizeByScale(2)
        self.conv_upsample1 = ConvNormRelu(self.channels[1], 3, activation='selu')
        self.conv_upsample2 = ConvNormRelu(self.channels[0], 3, activation='selu')
        self.conv_upsample3 = ConvNormRelu(self.channels[0], 3, activation='selu')
        self.conv_upsample4 = ConvNormRelu(self.channels[2], 3, activation='selu')
        self.conv_upsample5 = ConvNormRelu(sum(self.channels[1:]), 3, activation='selu')
        self.conv_concat2 = ConvNormRelu(sum(self.channels[1:]), 3, activation='selu')
        self.conv_concat3 = ConvNormRelu(sum(self.channels), 3, activation='selu')

        self.ua = UnionAttention(self.confidence)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        e2, e3, e4 = inputs

        e4_2 = self.upsample(e4)
        e4_4 = self.upsample(e4_2)
        e3_2 = self.upsample(e3)

        e3_1 = self.conv_upsample1(e4_2) * e3
        e2_1 = self.conv_upsample2(e4_4) * self.conv_upsample3(e3_2) * e2

        e3_2 = tf.concat([e3_1, self.conv_upsample4(e4_2)], axis=-1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = tf.concat([e2_1, self.conv_upsample5(self.upsample(e3_2))], axis=-1)
        outputs = self.conv_concat3(e2_2)

        outputs = self.ua(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({'confidence': self.confidence})

        return config
