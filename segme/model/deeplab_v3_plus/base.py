from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .encoder import Encoder
from .decoder import Decoder
from ...common import HeadProjection, HeadActivation


@register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3PlusBase(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1802.02611.pdf """

    def __init__(
            self, classes, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, low_filters, decoder_filters,
            add_strides=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters
        self.add_strides = add_strides

    @shape_type_conversion
    def build(self, input_shape):
        self.enc = Encoder(
            self.bone_arch, self.bone_init, self.bone_train, self.aspp_filters, self.aspp_stride,
            add_strides=self.add_strides)
        self.dec = Decoder(self.low_filters, self.decoder_filters)
        self.proj = HeadProjection(self.classes)
        self.act = HeadActivation(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        low_feats, high_feats, *add_feats = self.enc(inputs)
        dec_feats = self.dec([low_feats, high_feats])

        outputs = self.proj(dec_feats)

        return (outputs, dec_feats, *add_feats)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        low_feats_shape, high_feats_shape, *add_feats_shapes = self.enc.compute_output_shape(input_shape)
        dec_feats_shape = self.dec.compute_output_shape([low_feats_shape, high_feats_shape])
        outputs_shape = self.proj.compute_output_shape(dec_feats_shape)

        return (outputs_shape, dec_feats_shape, *add_feats_shapes)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride,
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters,
            'add_strides': self.add_strides
        })

        return config
