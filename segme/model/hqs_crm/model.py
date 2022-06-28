import tensorflow as tf
from keras import backend, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .decoder import Decoder
from .encoder import Encoder
from ...common import HeadActivation


@register_keras_serializable(package='SegMe>HqsCrm')
class HqsCrm(layers.Layer):
    def __init__(self, aspp_filters, aspp_drop, mlp_units, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # mask
            layers.InputSpec(ndim=4, axes={-1: 2}, dtype='float32'),  # coord
            layers.InputSpec(ndim=4, axes={-1: 2}, dtype='float32')  # cell
        ]

        self.aspp_filters = aspp_filters
        self.aspp_drop = aspp_drop
        self.mlp_units = mlp_units

    @shape_type_conversion
    def build(self, input_shape):
        self.encoder = Encoder()
        self.decoder = Decoder(self.aspp_filters, self.aspp_drop, self.mlp_units)
        self.act = HeadActivation(1)

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        images, masks, coords, cells = inputs

        feats2, feats4, feats32 = self.encoder([images, masks])
        logits = self.decoder([feats2, feats4, feats32, coords, cells])
        outputs = self.act(logits)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (1,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'aspp_filters': self.aspp_filters,
            'aspp_drop': self.aspp_drop,
            'mlp_units': self.mlp_units
        })

        return config


def build_hqs_crm(aspp_filters=(64, 64, 128), aspp_drop=0.5, mlp_units=(32, 32, 32, 32)):
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='mask', shape=[None, None, 1], dtype='uint8'),
        layers.Input(name='coord', shape=[None, None, 2], dtype='float32'),
        layers.Input(name='cell', shape=[None, None, 2], dtype='float32'),
    ]
    outputs = HqsCrm(aspp_filters, aspp_drop, mlp_units)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='hqs_crm')

    return model