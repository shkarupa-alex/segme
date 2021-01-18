from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .atsepconv import AtrousSepConv
from ...common import resize_by_sample, ConvBnRelu


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class Decoder(layers.Layer):
    def __init__(self, low_filters, decoder_filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # low level features
            layers.InputSpec(ndim=4)  # high level features
        ]
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters

    @shape_type_conversion
    def build(self, input_shape):
        self.proj = ConvBnRelu(self.low_filters, 1, use_bias=False)
        self.conv0 = AtrousSepConv(self.decoder_filters)
        self.conv1 = AtrousSepConv(self.decoder_filters)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        low_feats, high_feats = inputs

        outputs = resize_by_sample([high_feats, low_feats])
        outputs = layers.concatenate([self.proj(low_feats), outputs])
        outputs = self.conv0(outputs)
        outputs = self.conv1(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        low_shape, _ = input_shape

        return low_shape[:-1] + (self.decoder_filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters
        })

        return config
