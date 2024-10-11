from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.ops import fixed_conv
from segme.ops import fixed_depthwise_conv
from segme.ops import l2_normalize
from segme.policy.registry import LayerRegistry

CONVOLUTIONS = LayerRegistry()
CONVOLUTIONS.register("stdconv")(
    {
        "class_name": "SegMe>Policy>Conv>FixedConv",
        "config": {
            "kernel_constraint": {
                "class_name": "SegMe>Policy>Conv>StandardizedConstraint",
                "config": {"axes": [0, 1, 2]},
            }
        },
    }
)
CONVOLUTIONS.register("dw_stdconv")(
    {
        "class_name": "SegMe>Policy>Conv>FixedDepthwiseConv",
        "config": {
            "kernel_constraint": {
                "class_name": "SegMe>Policy>Conv>StandardizedConstraint",
                "config": {"axes": [0, 1]},
            }
        },
    }
)


@CONVOLUTIONS.register("conv")
@register_keras_serializable(package="SegMe>Policy>Conv")
class FixedConv(layers.Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        kwargs.pop("groups", None)
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def convolution_op(self, inputs, kernel):
        return fixed_conv(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def get_config(self):
        config = super().get_config()
        del config["groups"]

        return config


@CONVOLUTIONS.register("dw_conv")
@register_keras_serializable(package="SegMe>Policy>Conv")
class FixedDepthwiseConv(layers.DepthwiseConv2D):
    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        kwargs.pop("depth_multiplier", None)
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=1,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

    def _conv_op(self, inputs, kernel):
        return fixed_depthwise_conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def call(self, inputs):
        input_channel = self._get_input_channel(inputs.shape)
        outputs = self._conv_op(inputs, self.kernel)

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (
                    self.depth_multiplier * input_channel,
                )
            else:
                bias_shape = (1, self.depth_multiplier * input_channel) + (
                    1,
                ) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": config["depthwise_initializer"],
                "kernel_regularizer": config["depthwise_regularizer"],
                "kernel_constraint": config["depthwise_constraint"],
            }
        )

        del config["depth_multiplier"]
        del config["depthwise_initializer"]
        del config["depthwise_regularizer"]
        del config["depthwise_constraint"]

        return config


@register_keras_serializable(package="SegMe>Policy>Conv")
class StandardizedConstraint(constraints.Constraint):
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, w):
        w = backend.convert_to_tensor(w)

        if 4 != ops.ndim(w):
            raise ValueError(
                f"Expecting weight rank to equals 4, got {w.shape}"
            )

        mean, var = ops.moments(w, axes=self.axes, keepdims=True)
        w = ops.batch_normalization(w, mean, var, -1, epsilon=1.001e-5)

        return w

    def get_config(self):
        return {"axes": self.axes}


@CONVOLUTIONS.register("snconv")
@register_keras_serializable(package="SegMe>Policy>Conv")
class SpectralConv(FixedConv):
    """Implements https://arxiv.org/abs/1802.05957"""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        power_iterations=1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if power_iterations < 1:
            raise ValueError("Number of iterations must be greater then 0.")
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.u = self.add_weight(
            shape=(1, self.filters),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="u",
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs, training=False):
        if training:
            kernel, u = self.kernel, self.u
            w = ops.reshape(kernel, [-1, self.filters])

            for _ in range(self.power_iterations):
                v = l2_normalize(ops.matmul(u, ops.moveaxis(w, -1, -2)))
                u = l2_normalize(ops.matmul(v, w))

            sigma = ops.matmul(ops.matmul(v, w), ops.moveaxis(u, -1, -2))
            kernel = ops.reshape(kernel / sigma, self.kernel.shape)

            kernel = ops.stop_gradient(kernel)
            u = ops.stop_gradient(u)

            self.kernel.assign(kernel)
            self.u.assign(u)

        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"power_iterations": self.power_iterations})

        return config


@CONVOLUTIONS.register("dw_snconv")
@register_keras_serializable(package="SegMe>Policy>Conv")
class SpectralDepthwiseConv(FixedDepthwiseConv):
    """Implements https://arxiv.org/abs/1802.05957"""

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        power_iterations=1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if power_iterations < 1:
            raise ValueError("Number of iterations must be greater then 0.")
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        self.u = self.add_weight(
            shape=(1, self.channels),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="u",
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs, training=False):
        if training:
            kernel, u = self.kernel, self.u
            w = ops.reshape(kernel, [-1, self.channels])

            for _ in range(self.power_iterations):
                v = l2_normalize(ops.matmul(u, ops.moveaxis(w, -1, -2)))
                u = l2_normalize(ops.matmul(v, w))

            sigma = ops.matmul(ops.matmul(v, w), ops.moveaxis(u, -1, -2))
            kernel = ops.reshape(kernel / sigma, self.kernel.shape)

            kernel = ops.stop_gradient(kernel)
            u = ops.stop_gradient(u)

            self.kernel.assign(kernel)
            self.u.assign(u)

        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"power_iterations": self.power_iterations})

        return config
