from tensorflow.keras import Model, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .encoder import Encoder
from .decoder import Decoder
from ...common import HeadProjection, HeadActivation, resize_by_sample


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3Plus(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1802.02611.pdf """

    def __init__(
            self, classes, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, low_filters, decoder_filters,
            **kwargs):
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

    @shape_type_conversion
    def build(self, input_shape):
        self.enc = Encoder(self.bone_arch, self.bone_init, self.bone_train, self.aspp_filters, self.aspp_stride)
        self.dec = Decoder(self.low_filters, self.decoder_filters)
        self.proj = HeadProjection(self.classes)
        self.act = HeadActivation(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        low_feats, high_feats = self.enc(inputs)
        outputs = self.dec([low_feats, high_feats])
        outputs = self.proj(outputs)
        outputs = resize_by_sample([outputs, inputs])
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.proj.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature):
        proj_signature = self.proj.compute_output_signature(input_signature)

        return self.act.compute_output_signature(proj_signature)

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
            'decoder_filters': self.decoder_filters
        })

        return config


def build_deeplab_v3_plus(
        classes, bone_arch, bone_init, bone_train, aspp_filters=256, aspp_stride=16, low_filters=48,
        decoder_filters=256):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = DeepLabV3Plus(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
        aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='deeplab_v3_plus')

    return model
