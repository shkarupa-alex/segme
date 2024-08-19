import tensorflow as tf
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.backend.tensorflow.nn import _convert_data_format
from keras.src.saving import register_keras_serializable

from segme.policy.registry import LayerRegistry

CONVOLUTIONS = LayerRegistry()


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

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                f"`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

        self._tf_data_format = _convert_data_format(
            self.data_format, self.rank + 2
        )

    def convolution_op(self, inputs, kernel):
        paddings = "VALID" if "same" != self.padding else "SAME"

        if (
            "SAME" == paddings
            and max(self.kernel_size) > 1
            and max(self.strides) > 1
        ):
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)

            pad_hb = min(
                pad_h // 2, max(0, self.kernel_size[0] - self.strides[0])
            )
            pad_wb = min(
                pad_w // 2, max(0, self.kernel_size[1] - self.strides[1])
            )

            paddings = (
                (0, 0),
                (pad_hb, pad_h - pad_hb),
                (pad_wb, pad_w - pad_wb),
            )
            paddings = (
                ((0, 0),) + paddings
                if self.data_format == "channels_first"
                else paddings + ((0, 0),)
            )

        if (
            not tf.config.list_physical_devices("GPU")
            and max(self.dilation_rate) > 1
        ):
            # Current libxsmm and customized CPU implementations do not yet
            # support dilation rates > 1
            return tf.nn.convolution(
                inputs,
                kernel,
                strides=self.strides,
                padding=paddings,
                dilations=self.dilation_rate,
                data_format=self._tf_data_format,
                name=self.__class__.__name__,
            )

        return tf.nn.conv2d(
            inputs,
            kernel,
            strides=self.strides,
            padding=paddings,
            dilations=self.dilation_rate,
            data_format=self._tf_data_format,
            name=self.__class__.__name__,
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

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                f"`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

        self._tf_data_format = _convert_data_format(
            self.data_format, self.rank + 2
        )

    def _conv_op(self, inputs, kernel):
        strides = (
            (1, 1) + self.strides
            if self.data_format == "channels_first"
            else (1,) + self.strides + (1,)
        )
        paddings = "VALID" if "same" != self.padding else "SAME"

        if (
            "SAME" == paddings
            and max(self.kernel_size) > 1
            and max(self.strides) > 1
        ):
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)

            pad_hb = min(
                pad_h // 2, max(0, self.kernel_size[0] - self.strides[0])
            )
            pad_wb = min(
                pad_w // 2, max(0, self.kernel_size[1] - self.strides[1])
            )

            paddings = (
                (0, 0),
                (pad_hb, pad_h - pad_hb),
                (pad_wb, pad_w - pad_wb),
            )
            paddings = (
                ((0, 0),) + paddings
                if self.data_format == "channels_first"
                else paddings + ((0, 0),)
            )

        return tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides=strides,
            padding=paddings,
            dilations=self.dilation_rate,
            data_format=self._tf_data_format,
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


@CONVOLUTIONS.register("stdconv")
@register_keras_serializable(package="SegMe>Policy>Conv")
class StandardizedConv(FixedConv):
    """Implements https://arxiv.org/abs/1903.10520"""

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

    def _standardize_kernel(self, kernel, dtype=None):
        kernel = tf.cast(kernel, "float32")
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        kernel = tf.nn.batch_normalization(kernel, mean, var, None, None, 1e-5)
        kernel = tf.cast(kernel, dtype or self.compute_dtype)

        return kernel

    def convolution_op(self, inputs, kernel):
        kernel = self._standardize_kernel(kernel)

        return super().convolution_op(inputs, kernel)


@CONVOLUTIONS.register("dw_stdconv")
@register_keras_serializable(package="SegMe>Policy>Conv")
class StandardizedDepthwiseConv(FixedDepthwiseConv):
    """Implements https://arxiv.org/abs/1903.10520"""

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

    def _standardize_kernel(self, kernel, dtype=None):
        kernel = tf.cast(kernel, "float32")
        mean, var = tf.nn.moments(kernel, axes=[0, 1], keepdims=True)
        kernel = tf.nn.batch_normalization(kernel, mean, var, None, None, 1e-5)
        kernel = tf.cast(kernel, dtype or self.compute_dtype)

        return kernel

    def _conv_op(self, inputs, kernel):
        kernel = self._standardize_kernel(kernel)

        return super()._conv_op(inputs, kernel)


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
            name="sn_u",
            dtype=self.dtype,
        )

        super().build(input_shape)

    def before_train(self, inputs):
        kernel = tf.cast(self.kernel, self.dtype)
        u = tf.cast(self.u, self.dtype)

        w = tf.reshape(kernel, [-1, self.filters])

        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w))

        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
        kernel = tf.reshape(kernel / sigma, self.kernel.shape)
        kernel = tf.stop_gradient(kernel)
        u = tf.stop_gradient(u)

        kernel_update = self.kernel.value.assign(kernel, read_value=False)
        u_update = self.u.value.assign(u, read_value=False)
        with tf.control_dependencies([kernel_update, u_update]):
            outputs = tf.identity(inputs)

        return outputs

    def call(self, inputs, training=False):
        if training:
            outputs = self.before_train(inputs)
        else:
            outputs = inputs

        outputs = super().call(outputs)

        return outputs

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
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        power_iterations=1,
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
            name="sn_u",
            dtype=self.dtype,
        )

        super().build(input_shape)

    def before_train(self, inputs):
        kernel = tf.cast(self.kernel, self.dtype)
        u = tf.cast(self.u, self.dtype)

        w = tf.reshape(kernel, [-1, self.channels])

        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w))

        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
        kernel = tf.reshape(kernel / sigma, self.kernel.shape)
        kernel = tf.stop_gradient(kernel)
        u = tf.stop_gradient(u)

        kernel_update = self.kernel.value.assign(kernel, read_value=False)
        u_update = self.u.value.assign(u, read_value=False)
        with tf.control_dependencies([kernel_update, u_update]):
            outputs = tf.identity(inputs)

        return outputs

    def call(self, inputs, training=False):
        if training:
            outputs = self.before_train(inputs)
        else:
            outputs = inputs

        outputs = super().call(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"power_iterations": self.power_iterations})

        return config
