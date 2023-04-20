from keras import Model, layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.model.segmentation.deeplab_v3_plus.base import DeepLabV3PlusBase


@register_keras_serializable(package='SegMe>Model>Segmentation>DeepLabV3Plus')
class DeepLabV3Plus(DeepLabV3PlusBase):
    """ Reference: https://arxiv.org/pdf/1802.02611.pdf """

    def call(self, inputs, **kwargs):
        outputs = super().call(inputs)
        outputs = self.resize([outputs, inputs])
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.classes,)

    def compute_output_signature(self, input_signature):
        proj_signature = super().compute_output_signature(input_signature)

        return self.act.compute_output_signature(proj_signature)


def build_deeplab_v3_plus(classes, aspp_filters=256, aspp_stride=32, low_filters=48, decoder_filters=256):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = DeepLabV3Plus(
        classes, aspp_filters=aspp_filters, aspp_stride=aspp_stride, low_filters=low_filters,
        decoder_filters=decoder_filters)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='deeplab_v3_plus')

    return model
