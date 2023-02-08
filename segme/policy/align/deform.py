import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfmiss.keras.layers import DCNv2
from segme.common.sequent import Sequential
from segme.common.resize import BilinearInterpolation


@register_keras_serializable(package='SegMe>Policy>Align>Deformable')
class DeformableFeatureAlignment(layers.Layer):
    """
    Proposed in "FaPN: Feature-aligned Pyramid Network for Dense Image Prediction"
    https://arxiv.org/pdf/2108.07058.pdf
    """

    def __init__(self, filters, deformable_groups=8, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.filters = filters
        self.deformable_groups = deformable_groups

    @shape_type_conversion
    def build(self, input_shape):
        self.interpolate = BilinearInterpolation(None)
        self.select = FeatureSelection(self.filters)
        self.offset = layers.Conv2D(self.filters * 2, 1, use_bias=False, kernel_initializer='he_uniform')
        self.dcn = DCNv2(
            self.filters, 3, padding='same', deformable_groups=self.deformable_groups, custom_alignment=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs
        coarse = self.interpolate([coarse, fine])

        fine_calibrated = self.select(fine)

        fine_coarse = layers.concatenate([fine_calibrated, coarse * 2.])
        align_offset = self.offset(fine_coarse)
        coarse_aligned = self.dcn([coarse, align_offset])
        coarse_aligned = tf.nn.relu(coarse_aligned)

        outputs = coarse_aligned + fine_calibrated

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'deformable_groups': self.deformable_groups
        })

        return config


@register_keras_serializable(package='SegMe>Policy>Align>Deformable')
class FeatureSelection(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.attend = Sequential([
            layers.GlobalAvgPool2D(keepdims=True),
            layers.Conv2D(channels, 1, activation='sigmoid', use_bias=False, kernel_initializer='he_uniform'),
        ])
        self.conv = layers.Conv2D(self.filters, 1, use_bias=False, kernel_initializer='he_uniform')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        attention = self.attend(inputs)
        outputs = inputs * (attention + 1)  # same as inputs * attention + inputs
        outputs = self.conv(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
