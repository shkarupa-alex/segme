import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.aspp import AtrousSpatialPyramidPooling
from segme.common.convnormact import ConvNormAct
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package='SegMe>Model>Segmentation>DeepLabV3Plus')
class Decoder(layers.Layer):
    def __init__(self, aspp_filters, aspp_stride, low_filters, decoder_filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters

    @shape_type_conversion
    def build(self, input_shape):
        self.aspp = AtrousSpatialPyramidPooling(self.aspp_filters, self.aspp_stride)
        self.resize = BilinearInterpolation(None)
        self.fineproj = ConvNormAct(self.low_filters, 1)
        self.outproj = Sequence([
            ConvNormAct(None, 3), ConvNormAct(self.decoder_filters, 1),
            ConvNormAct(None, 3), ConvNormAct(self.decoder_filters, 1)
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        coarse = self.aspp(coarse)
        coarse = self.resize([coarse, fine])
        fine = self.fineproj(fine)

        outputs = tf.concat([fine, coarse], axis=-1)
        outputs = self.outproj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.decoder_filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride,
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters
        })

        return config
