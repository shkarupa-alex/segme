import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.model.uper_net.decoder import Decoder
from segme.model.uper_net.head import Head
from segme.common.backbone import Backbone


@register_keras_serializable(package='SegMe>Model>UPerNet')
class UPerNet(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1807.10221v1.pdf """

    def __init__(self, classes, dropout, dec_filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes
        self.dropout = dropout
        self.dec_filters = dec_filters

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(scales=[4, 8, 16, 32])
        self.decode = Decoder(self.dec_filters)
        self.head = Head(self.classes, self.dropout)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats = self.bone(inputs)
        outputs = self.decode(feats)
        outputs = self.head([outputs, inputs])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.classes,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'dropout': self.dropout,
            'dec_filters': self.dec_filters
        })

        return config


def build_uper_net(classes, dropout=0.1, dec_filters=512):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = UPerNet(classes, dropout=dropout, dec_filters=dec_filters)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='uper_net')

    return model
