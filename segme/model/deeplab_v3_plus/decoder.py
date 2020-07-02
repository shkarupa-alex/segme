from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import AtrousSepConv2D, up_by_sample_2d


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class Decoder(layers.Layer):
    def __init__(self, low_filters, decoder_filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, dtype='uint8'),  # images
            layers.InputSpec(ndim=4),  # low level features
            layers.InputSpec(ndim=4)  # high level features
        ]
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters

    @shape_type_conversion
    def build(self, input_shape):
        self.proj = Sequential([
            layers.Conv2D(self.low_filters, 1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv0 = AtrousSepConv2D(self.decoder_filters)
        self.conv1 = AtrousSepConv2D(self.decoder_filters)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        images, low_feats, high_feats = inputs

        outputs = up_by_sample_2d([high_feats, low_feats])
        outputs = layers.concatenate([self.proj(low_feats), outputs])
        outputs = self.conv0(outputs)
        outputs = self.conv1(outputs)
        outputs = up_by_sample_2d([outputs, images])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        images_shape, _, _ = input_shape

        return images_shape[:-1] + (self.decoder_filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters
        })

        return config
