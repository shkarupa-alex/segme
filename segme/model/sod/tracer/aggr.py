import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.resize import BilinearInterpolation
from segme.model.sod.tracer.uniatt import UnionAttention


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
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

        self.upsample2 = BilinearInterpolation(2)
        self.upsample4 = BilinearInterpolation(4)
        self.conv_upsample1 = ConvNormAct(self.channels[1], 3)
        self.conv_upsample2 = ConvNormAct(self.channels[0], 3)
        self.conv_upsample3 = ConvNormAct(self.channels[0], 3)
        self.conv_upsample4 = ConvNormAct(self.channels[2], 3)
        self.conv_upsample5 = ConvNormAct(sum(self.channels[1:]), 3)
        self.conv_concat2 = ConvNormAct(sum(self.channels[1:]), 3)
        self.conv_concat3 = ConvNormAct(sum(self.channels), 3)

        self.ua = UnionAttention(self.confidence)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        e2, e3, e4 = inputs

        e3_x2 = self.upsample2(e3)
        e4_x2 = self.upsample2(e4)
        e4_x4 = self.upsample4(e4)

        e3_1 = self.conv_upsample1(e4_x2) * e3
        e2_1 = self.conv_upsample2(e4_x4) * self.conv_upsample3(e3_x2) * e2

        e3_2 = tf.concat([e3_1, self.conv_upsample4(e4_x2)], axis=-1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = tf.concat([e2_1, self.conv_upsample5(self.upsample2(e3_2))], axis=-1)
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
