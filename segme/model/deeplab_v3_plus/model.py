from keras import Model, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .base import DeepLabV3PlusBase
from ...common import resize_by_sample


@register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3Plus(DeepLabV3PlusBase):
    """ Reference: https://arxiv.org/pdf/1802.02611.pdf """

    def __init__(self, *args, **kwargs):
        kwargs.pop('add_strides', None)
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        outputs = super().call(inputs)[0]
        outputs = resize_by_sample([outputs, inputs])
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + super().compute_output_shape(input_shape)[0][-1:]

        return output_shape

    def compute_output_signature(self, input_signature):
        proj_signature = super().compute_output_signature(input_signature)

        return self.act.compute_output_signature(proj_signature)


def build_deeplab_v3_plus(
        classes, bone_arch, bone_init, bone_train, aspp_filters=256, aspp_stride=32, low_filters=48,
        decoder_filters=256):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = DeepLabV3Plus(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
        aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='deeplab_v3_plus')

    return model
