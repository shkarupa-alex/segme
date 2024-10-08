from keras.src import constraints
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import regularizers
from keras.src.saving import register_keras_serializable

from segme.policy.registry import LayerRegistry

ACTIVATIONS = LayerRegistry()
ACTIVATIONS.register("relu")(layers.ReLU)
ACTIVATIONS.register("leakyrelu")(layers.LeakyReLU)
ACTIVATIONS.register("gelu")(
    {"class_name": "Activation", "config": {"activation": "gelu"}}
)
ACTIVATIONS.register("selu")(
    {"class_name": "Activation", "config": {"activation": "selu"}}
)
ACTIVATIONS.register("silu")(
    {"class_name": "Activation", "config": {"activation": "silu"}}
)


@ACTIVATIONS.register("tlu")
@register_keras_serializable(package="SegMe>Policy>Activation")
class TLU(layers.Layer):
    """Implements https://arxiv.org/abs/1911.09737"""

    def __init__(
        self,
        tau_initializer="zeros",
        tau_regularizer=None,
        tau_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.tau_initializer = initializers.get(tau_initializer)
        self.tau_regularizer = regularizers.get(tau_regularizer)
        self.tau_constraint = constraints.get(tau_constraint)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        tau_shape = [1] * (len(input_shape) - 1) + [channels]
        self.tau = self.add_weight(
            name="tau",
            shape=tau_shape,
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return ops.maximum(inputs, self.tau)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tau_initializer": initializers.serialize(self.tau_initializer),
                "tau_regularizer": regularizers.serialize(self.tau_regularizer),
                "tau_constraint": constraints.serialize(self.tau_constraint),
            }
        )

        return config


@ACTIVATIONS.register("approxgelu")
@register_keras_serializable(package="SegMe>Policy>Activation")
class ApproximateGELU(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        return ops.gelu(inputs, approximate=True)

    def compute_output_shape(self, input_shape):
        return input_shape
