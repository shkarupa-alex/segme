from keras.src import initializers
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from keras.src.utils.argument_validation import standardize_tuple

from segme.common.convnormact import ConvNorm
from segme.common.convnormact import ConvNormAct
from segme.common.drop import DropPath
from segme.common.se import SE


@register_keras_serializable(package="SegMe>Common")
class MBConv(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        fused,
        strides=1,
        expand_ratio=4.0,
        se_ratio=0.25,
        gamma_initializer="ones",
        drop_ratio=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=1)

        self.filters = filters
        self.kernel_size = kernel_size
        self.fused = fused
        self.strides = standardize_tuple(strides, 2, "strides")
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.drop_ratio = drop_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        self.with_expansion = 1.0 != self.expand_ratio
        self.with_se = 0.0 != self.se_ratio
        self.with_residual = 1 == max(self.strides) and channels == self.filters

        expand_filters = int(channels * self.expand_ratio)
        kernel_initializer = {
            "class_name": "VarianceScaling",
            "config": {
                "scale": 2.0,
                "mode": "fan_out",
                "distribution": "untruncated_normal",
            },
        }

        current_shape = input_shape
        if self.with_expansion and self.fused:
            self.expand = ConvNormAct(
                expand_filters,
                self.kernel_size,  # TODO  if fused
                strides=self.strides,
                kernel_initializer=kernel_initializer,
                name="expand",
                dtype=self.dtype_policy,
            )
        elif self.with_expansion:
            self.expand = ConvNormAct(
                expand_filters,
                1,
                kernel_initializer=kernel_initializer,
                name="expand",
                dtype=self.dtype_policy,
            )
        self.expand.build(current_shape)
        current_shape = self.expand.compute_output_shape(current_shape)

        if not self.fused:
            self.dwconv = ConvNormAct(
                None,
                self.kernel_size,
                strides=self.strides,
                kernel_initializer=kernel_initializer,
                name="dw",
                dtype=self.dtype_policy,
            )
            self.dwconv.build(current_shape)
            current_shape = self.dwconv.compute_output_shape(current_shape)

        if self.with_se:
            self.se = SE(self.se_ratio, name="se", dtype=self.dtype_policy)
            self.se.build(current_shape)

        if self.fused and not self.with_expansion:
            self.proj = ConvNormAct(
                self.filters,
                self.kernel_size,
                strides=self.strides,
                kernel_initializer=kernel_initializer,
                gamma_initializer=self.gamma_initializer,
                name="proj",
                dtype=self.dtype_policy,
            )
        else:
            self.proj = ConvNorm(
                self.filters,
                1,
                kernel_initializer=kernel_initializer,
                gamma_initializer=self.gamma_initializer,
                name="proj",
                dtype=self.dtype_policy,
            )
        self.proj.build(current_shape)

        if self.with_residual:
            self.drop = DropPath(
                self.drop_ratio, name="drop", dtype=self.dtype_policy
            )
        elif 0.0 != self.drop_ratio:
            raise ValueError(
                "Dropout ration is set, but residual branch can't be executed."
            )

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        outputs = inputs

        if self.with_expansion:
            outputs = self.expand(outputs)

        if not self.fused:
            outputs = self.dwconv(outputs)

        if self.with_se:
            outputs = self.se(outputs)

        outputs = self.proj(outputs)

        if self.with_residual:
            outputs = self.drop(outputs) + inputs

        return outputs

    def compute_output_shape(self, input_shape):
        if self.fused and self.with_expansion:
            output_shape = self.expand.compute_output_shape(input_shape)
        elif self.fused:
            output_shape = self.proj.compute_output_shape(input_shape)
        else:
            output_shape = self.dwconv.compute_output_shape(input_shape)

        return output_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "fused": self.fused,
                "strides": self.strides,
                "expand_ratio": self.expand_ratio,
                "se_ratio": self.se_ratio,
                "gamma_initializer": initializers.serialize(
                    self.gamma_initializer
                ),
                "drop_ratio": self.drop_ratio,
            }
        )

        return config
