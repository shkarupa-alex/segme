import numpy as np
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from keras.src.utils.argument_validation import standardize_tuple

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence
from segme.ops import modulated_deformable_column


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


@register_keras_serializable(package="SegMe>Policy>Align>Deformable")
class DCNv2(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        dilation_rate=(1, 1),
        deformable_groups=1,
        use_bias=True,
        custom_alignment=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_spec = InputSpec(ndim=4)  # inputs
        if custom_alignment:
            self.input_spec = [
                InputSpec(ndim=4),  # inputs
                InputSpec(ndim=4),  # alignments
            ]

        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
        self.strides = standardize_tuple(strides, 2, "strides")
        self.padding = padding
        self.dilation_rate = standardize_tuple(
            dilation_rate, 2, "dilation_rate"
        )
        self.deformable_groups = deformable_groups
        self.use_bias = use_bias
        self.custom_alignment = custom_alignment

        if "valid" == str(self.padding).lower():
            self._padding = (0, 0, 0, 0)
        elif "same" == str(self.padding).lower():
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)
            self._padding = (
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        else:
            raise ValueError(
                f"The `padding` argument must be one of `valid` or `same`. "
                f"Received: {padding}"
            )

    def build(self, input_shape):
        channels = input_shape[-1]
        if self.custom_alignment:
            channels = input_shape[0][-1]

        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        if channels < self.deformable_groups:
            raise ValueError(
                "Number of deformable groups should be less or "
                "equals to channel dimension size"
            )

        kernel_shape = (
            self.kernel_size[0] * self.kernel_size[1] * channels,
            self.filters,
        )
        kernel_stdv = 1.0 / np.sqrt(np.prod((channels,) + self.kernel_size))
        kernel_init = initializers.RandomUniform(-kernel_stdv, kernel_stdv)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=kernel_init,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
                dtype=self.dtype,
            )

        self.offset_size = (
            self.deformable_groups
            * 2
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        self.offset_mask = layers.Conv2D(
            self.offset_size * 3 // 2,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            kernel_initializer="zeros",
            dtype=self.dtype_policy,
        )
        if self.custom_alignment:
            self.offset_mask.build(input_shape[1])
        else:
            self.offset_mask.build(input_shape)

        self.sigmoid = layers.Activation("sigmoid", dtype=self.dtype_policy)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        alignments = inputs
        if self.custom_alignment:
            inputs, alignments = inputs
            alignments = ops.cast(alignments, inputs.dtype)

        offset_mask = self.offset_mask(alignments)

        offset, mask = (
            offset_mask[..., : self.offset_size],
            offset_mask[..., self.offset_size :],
        )
        mask = self.sigmoid(mask) * 2.0  # (0.; 2.) with mean == 1.

        columns = modulated_deformable_column(
            inputs,
            offset,
            mask,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self._padding,
            dilation_rate=self.dilation_rate,
            deformable_groups=self.deformable_groups,
        )

        outputs = ops.matmul(columns, self.kernel)
        out_shape = ops.shape(offset_mask)[:-1] + (self.filters,)

        outputs = ops.reshape(outputs, out_shape)
        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        return outputs

    def compute_output_shape(self, input_shape):
        source_shape = input_shape
        if self.custom_alignment:
            source_shape = input_shape[1]

        offset_mask_shape = self.offset_mask.compute_output_shape(source_shape)

        return offset_mask_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "deformable_groups": self.deformable_groups,
                "use_bias": self.use_bias,
                "custom_alignment": self.custom_alignment,
            }
        )

        return config
