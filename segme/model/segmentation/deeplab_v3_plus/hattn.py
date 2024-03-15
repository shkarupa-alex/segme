from tf_keras import layers, models
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.hmsattn import HierarchicalMultiScaleAttention
from segme.model.segmentation.deeplab_v3_plus.base import DeepLabV3PlusBase


@register_keras_serializable(package='SegMe>Model>Segmentation>DeepLabV3Plus')
class DeepLabV3PlusWithHierarchicalAttention(layers.Layer):
    """ Reference: https://arxiv.org/pdf/2005.10821.pdf """

    def __init__(self, scales, classes, aspp_filters, aspp_stride, low_filters, decoder_filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        self.scales = scales
        self.classes = classes
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters

    @shape_type_conversion
    def build(self, input_shape):
        self.deeplab = DeepLabV3PlusBase(
            self.classes, self.aspp_filters, self.aspp_stride, self.low_filters, self.decoder_filters)
        self.deeplab._return_decfeat = True

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
        classes, aspp_filters=256, aspp_stride=32, low_filters=48, decoder_filters=256,
        scales=((0.5,), (0.25, 0.5, 2.0))):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = DeepLabV3PlusWithHierarchicalAttention(
        scales=scales, classes=classes, aspp_filters=aspp_filters, aspp_stride=aspp_stride, low_filters=low_filters,
        decoder_filters=decoder_filters)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='deeplab_v3_plus_with_hierarchical_attention')

    return model
