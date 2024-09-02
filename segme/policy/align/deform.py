from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from tfmiss.keras.layers import DCNv2

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Policy>Align>Deformable")
class DeformableFeatureAlignment(layers.Layer):
    """
    Proposed in "FaPN: Feature-aligned Pyramid Network for Dense Image
    Prediction"
    https://arxiv.org/pdf/2108.07058
    """

    def __init__(self, filters, deformable_groups=8, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # fine
            InputSpec(ndim=4),
        ]  # coarse

        self.filters = filters
        self.deformable_groups = deformable_groups

    def build(self, input_shape):
        self.interpolate = BilinearInterpolation(dtype=self.dtype_policy)

        self.select = FeatureSelection(self.filters, dtype=self.dtype_policy)
        self.select.build(input_shape[0])

        self.offset = Conv(
            self.filters * 2,
            1,
            use_bias=False,
            kernel_initializer="he_uniform",
            dtype=self.dtype_policy,
        )
        self.offset.build(
            input_shape[0][:-1] + (self.filters + input_shape[1][-1],)
        )

        self.dcn = DCNv2(
            self.filters,
            3,
            padding="same",
            deformable_groups=self.deformable_groups,
            custom_alignment=True,
            dtype=self.dtype_policy,
        )
        self.dcn.build(
            [input_shape[1], input_shape[0][:-1] + (self.filters * 2,)]
        )

        self.act = Act(dtype=self.dtype_policy)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs
        coarse = self.interpolate([coarse, fine])

        fine_calibrated = self.select(fine)

        fine_coarse = layers.concatenate([fine_calibrated, coarse * 2.0])
        align_offset = self.offset(fine_coarse)
        coarse_aligned = self.dcn([coarse, align_offset])
        coarse_aligned = self.act(coarse_aligned)

        outputs = coarse_aligned + fine_calibrated

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "deformable_groups": self.deformable_groups,
            }
        )

        return config


@register_keras_serializable(package="SegMe>Policy>Align>Deformable")
class FeatureSelection(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.filters = filters

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=4, axes={-1: channels})

        self.attend = Sequence(
            [
                layers.GlobalAveragePooling2D(
                    keepdims=True, dtype=self.dtype_policy
                ),
                layers.Conv2D(
                    channels,
                    1,
                    activation="sigmoid",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    dtype=self.dtype_policy,
                ),
            ],
            dtype=self.dtype_policy,
        )
        self.attend.build(input_shape)

        self.conv = Conv(
            self.filters,
            1,
            use_bias=False,
            kernel_initializer="he_uniform",
            dtype=self.dtype_policy,
        )
        self.conv.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        attention = self.attend(inputs)
        outputs = inputs * (
            attention + 1.0
        )  # same as inputs * attention + inputs
        outputs = self.conv(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})

        return config
