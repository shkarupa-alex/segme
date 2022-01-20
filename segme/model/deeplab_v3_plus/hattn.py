from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .base import DeepLabV3PlusBase
from ...common import HierarchicalMultiScaleAttention


@register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3PlusWithHighLevelFeatures(DeepLabV3PlusBase):
    def call(self, inputs, **kwargs):
        outputs, dec_feats = super().call(inputs, **kwargs)[:2]

        return outputs, dec_feats

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        outputs_shape, dec_feats_shape = super().compute_output_shape(input_shape)[:2]

        return outputs_shape, dec_feats_shape


@register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3PlusWithHierarchicalAttention(layers.Layer):
    """ Reference: https://arxiv.org/pdf/2005.10821.pdf """

    def __init__(
            self, scales, classes, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, low_filters,
            decoder_filters, **kwargs):
        super().__init__(**kwargs)
        self.scales = scales
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
        self.deeplab = DeepLabV3PlusWithHighLevelFeatures(
            classes=self.classes,
            bone_arch=self.bone_arch,
            bone_init=self.bone_init,
            bone_train=self.bone_train,
            aspp_filters=self.aspp_filters,
            aspp_stride=self.aspp_stride,
            low_filters=self.low_filters,
            decoder_filters=self.decoder_filters)
        self.hmsattn = HierarchicalMultiScaleAttention(self.deeplab, self.scales)

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        logits = self.hmsattn(inputs)
        outputs = self.deeplab.act(logits)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'scales': self.scales,
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

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        outputs_shape = self.hmsattn.compute_output_shape(input_shape)

        return outputs_shape

    def compute_output_signature(self, input_signature):
        output_signature = self.hmsattn.compute_output_signature(input_signature)
        output_signature = self.deeplab.act.compute_output_signature(output_signature)

        return output_signature


def build_deeplab_v3_plus_with_hierarchical_attention(
        classes, bone_arch, bone_init, bone_train, aspp_filters=256, aspp_stride=32, low_filters=48,
        decoder_filters=256, scales=((0.5,), (0.25, 0.5, 2.0))):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = DeepLabV3PlusWithHierarchicalAttention(
        scales=scales, classes=classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train,
        aspp_filters=aspp_filters, aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters)(
        inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='deeplab_v3_plus_with_hierarchical_attention')

    return model
