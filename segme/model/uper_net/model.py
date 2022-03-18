import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.norm import LayerNorm
from .decoder import Decoder
from .head import Head
from ...backbone import Backbone
from ...common import ConvNormRelu


@register_keras_serializable(package='SegMe>UPerNet')
class UPerNet(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1807.10221v1.pdf """

    def __init__(
            self, classes, bone_arch, bone_init, bone_train, dropout, dec_filters, psp_sizes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes
        self.dropout = dropout
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train
        self.dec_filters = dec_filters
        self.psp_sizes = psp_sizes

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(self.bone_arch, self.bone_init, self.bone_train, scales=[4, 8, 16, 32])
        if self.bone_arch.startswith('swin_'):
            self.norm = [LayerNorm() for _ in range(4)]
        self.decode = Decoder(self.dec_filters, self.psp_sizes)
        self.head = Head(self.classes, self.dropout)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats = self.bone(inputs)
        if self.bone_arch.startswith('swin_'):
            feats = [norm(feat) for norm, feat in zip(self.norm, feats)]

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
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'dropout': self.dropout,
            'dec_filters': self.dec_filters,
            'psp_sizes': self.psp_sizes
        })

        return config


def build_uper_net(
        classes, bone_arch, bone_init, bone_train, dropout=0.1, dec_filters=512, psp_sizes=(1, 2, 3, 6)):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = UPerNet(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, dropout=dropout,
        dec_filters=dec_filters, psp_sizes=psp_sizes)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='uper_net')

    return model
