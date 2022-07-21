from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfmiss.keras.layers import DCNv2
from .resizebysample import resize_by_sample
from .sameconv import SameConv


@register_keras_serializable(package='SegMe')
class DeformableFeatureAlignment(layers.Layer):
    def __init__(self, filters, deformable_groups=8, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # coarse
            layers.InputSpec(ndim=4)  # fine
        ]

        self.filters = filters
        self.deformable_groups = deformable_groups

    @shape_type_conversion
    def build(self, input_shape):
        self.select = FeatureSelection(self.filters)
        self.offset = SameConv(
            self.filters * 2, 1, use_bias=False, kernel_initializer='he_uniform')
        self.dcn = DCNv2(
            self.filters, 3, padding='same', deformable_groups=self.deformable_groups, custom_alignment=True)
        self.relu = layers.ReLU()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        coarse, fine = inputs
        coarse = resize_by_sample([coarse, fine])

        fine_calibrated = self.select(fine)

        fine_coarse = layers.concatenate([fine_calibrated, coarse * 2.])
        align_offset = self.offset(fine_coarse)
        coarse_aligned = self.dcn([coarse, align_offset])
        coarse_aligned = self.relu(coarse_aligned)

        outputs = coarse_aligned + fine_calibrated

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'deformable_groups': self.deformable_groups
        })

        return config


@register_keras_serializable(package='SegMe')
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

        self.attend = models.Sequential([
            layers.GlobalAvgPool2D(keepdims=True),
            SameConv(channels, 1, activation='sigmoid', use_bias=False, kernel_initializer='he_uniform')
        ])
        self.conv = SameConv(self.filters, 1, use_bias=False, kernel_initializer='he_uniform')

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
