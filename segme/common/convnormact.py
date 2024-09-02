from keras.src import constraints
from keras.src import initializers
from keras.src import layers
from keras.src import regularizers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.policy import cnapol


def Conv(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    policy=None,
    **kwargs,
):
    policy = cnapol.deserialize(policy or cnapol.global_policy())

    kwargs.update(
        {
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": padding,
            "data_format": data_format,
            "dilation_rate": dilation_rate,
            "activation": activation,
            "use_bias": use_bias,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "kernel_regularizer": kernel_regularizer,
            "bias_regularizer": bias_regularizer,
            "kernel_constraint": kernel_constraint,
            "bias_constraint": bias_constraint,
        }
    )

    if filters is None:  # depthwise
        return cnapol.CONVOLUTIONS.new(f"dw_{policy.conv_type}", **kwargs)

    return cnapol.CONVOLUTIONS.new(policy.conv_type, filters=filters, **kwargs)


# TODO: more args?
def Norm(
    data_format=None,
    epsilon=None,
    gamma_initializer="ones",
    policy=None,
    **kwargs,
):
    policy = cnapol.deserialize(policy or cnapol.global_policy())

    kwargs.update(
        {"gamma_initializer": gamma_initializer, "data_format": data_format}
    )
    if epsilon is not None:
        kwargs["epsilon"] = epsilon

    return cnapol.NORMALIZATIONS.new(policy.norm_type, **kwargs)


def Act(policy=None, **kwargs):
    policy = cnapol.deserialize(policy or cnapol.global_policy())

    return cnapol.ACTIVATIONS.new(policy.act_type, **kwargs)


@register_keras_serializable(package="SegMe>Common>ConvNormAct")
class ConvNormAct(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        epsilon=None,
        gamma_initializer="ones",
        policy=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    def build(self, input_shape):
        self.conv = Conv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            policy=self.policy,
            name="policy_conv",
            dtype=self.dtype_policy,
        )

        self.norm = Norm(
            data_format=self.data_format,
            epsilon=self.epsilon,
            gamma_initializer=self.gamma_initializer,
            policy=self.policy,
            name="policy_norm",
            dtype=self.dtype_policy,
        )
        self.act = Act(
            policy=self.policy, name="policy_act", dtype=self.dtype_policy
        )

        current_shape = input_shape
        self.conv.build(current_shape)

        current_shape = self.conv.compute_output_shape(current_shape)
        self.norm.build(current_shape)
        self.act.build(current_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)
        outputs = self.act(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "epsilon": self.epsilon,
                "gamma_initializer": initializers.serialize(
                    self.gamma_initializer
                ),
                "policy": cnapol.serialize(self.policy),
            }
        )

        return config


@register_keras_serializable(package="SegMe>Common>ConvNormAct")
class ConvNorm(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        epsilon=None,
        gamma_initializer="ones",
        policy=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    def build(self, input_shape):
        self.conv = Conv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            policy=self.policy,
            name="policy_conv",
            dtype=self.dtype_policy,
        )
        self.norm = Norm(
            data_format=self.data_format,
            epsilon=self.epsilon,
            gamma_initializer=self.gamma_initializer,
            policy=self.policy,
            name="policy_norm",
            dtype=self.dtype_policy,
        )

        current_shape = input_shape
        self.conv.build(current_shape)

        current_shape = self.conv.compute_output_shape(current_shape)
        self.norm.build(current_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "epsilon": self.epsilon,
                "gamma_initializer": initializers.serialize(
                    self.gamma_initializer
                ),
                "policy": cnapol.serialize(self.policy),
            }
        )

        return config


@register_keras_serializable(package="SegMe>Common>ConvNormAct")
class ConvAct(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        policy=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    def build(self, input_shape):
        # TODO: after bn?
        if self.policy.act_type in {"relu", "leaky_relu"}:
            kernel_initializer = "he_uniform"
        else:
            kernel_initializer = self.kernel_initializer

        self.conv = Conv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            policy=self.policy,
            name="policy_conv",
            dtype=self.dtype_policy,
        )
        self.act = Act(
            policy=self.policy, name="policy_act", dtype=self.dtype_policy
        )

        current_shape = input_shape
        self.conv.build(current_shape)

        current_shape = self.conv.compute_output_shape(current_shape)
        self.act.build(current_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.act(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "policy": cnapol.serialize(self.policy),
            }
        )

        return config
