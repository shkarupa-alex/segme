import math

import tensorflow as tf
from keras.src import backend, constraints
from keras.src import initializers
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src import regularizers
from keras.src.saving import register_keras_serializable


from segme.common.shape import get_shape
from segme.policy.registry import LayerRegistry

NORMALIZATIONS = LayerRegistry()
NORMALIZATIONS.register("gn321em5")(
    {
        "class_name": "SegMe>Policy>Normalization>GroupNorm",
        "config": {"groups": 32, "epsilon": 1.001e-5},
    }
)
NORMALIZATIONS.register("ln1em5")(
    {
        "class_name": "SegMe>Policy>Normalization>LayerNorm",
        "config": {"epsilon": 1.001e-5},
    }
)

@NORMALIZATIONS.register("bn")
@register_keras_serializable(package="SegMe>Policy>Normalization")
class BatchNorm(layers.BatchNormalization):
    """Overload for data_format understanding"""

    def __init__(
        self,
        data_format=None,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        synchronized=False,
        **kwargs,
    ):
        self.data_format = backend.standardize_data_format(data_format)
        axis = -1 if "channels_last" == self.data_format else 1
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            synchronized=synchronized,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"data_format": self.data_format})

        del config["axis"]

        return config


@NORMALIZATIONS.register("ln")
@register_keras_serializable(package="SegMe>Policy>Normalization")
class LayerNorm(layers.LayerNormalization):
    """Overload for data_format understanding"""

    def __init__(
        self,
        data_format=None,
        epsilon=1e-3,
        center=True,
        scale=True,
        rms_scaling=False,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        self.data_format = backend.standardize_data_format(data_format)
        axis = -1 if "channels_last" == self.data_format else 1
        super().__init__(
            axis=axis,
            epsilon=epsilon,
            center=center,
            scale=scale,
            rms_scaling=rms_scaling,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"data_format": self.data_format})

        del config["axis"]

        return config


@NORMALIZATIONS.register("lwn")
@register_keras_serializable(package="SegMe>Policy>Normalization")
class LayerwiseNorm(LayerNorm):
    """Overload for exact paper implementation"""

    def build(self, input_shape):
        self.axis = list(range(1, len(input_shape)))
        super().build(input_shape)


@NORMALIZATIONS.register("gn")
@register_keras_serializable(package="SegMe>Policy>Normalization")
class GroupNorm(layers.GroupNormalization):
    # TODO
    """Overload for fused implementation, groups estimation and BCHW issue fix"""

    def __init__(
        self,
        groups=None,
        data_format=None,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        self.data_format = backend.standardize_data_format(data_format)
        axis = -1 if "channels_last" == self.data_format else 1
        super().__init__(
            groups=-1,
            axis=axis,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs,
        )
        self._groups = groups

    def build(self, input_shape):
        self.rank = len(input_shape)
        self.channels = input_shape[self.axis]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. Found `None`."
            )

        if self._groups is None:
            # Best results in paper obtained with groups=32 or channels_per_group=16
            best_cpg = self.channels // (
                2 ** math.floor(math.log2(self.channels) / 2)
            )
            best_grp = min(32, self.channels // min(16, best_cpg))
            if self.channels % best_grp:
                raise ValueError(
                    f"Unable to choose best number of groups for the number of channels ({self.channels}). "
                    f"It supposed to be somewhere near {best_grp}."
                )

            self.groups = best_grp
        elif -1 == self._groups:
            self.groups = self.channels
        else:
            self.groups = self._groups
            if self.channels < self.groups:
                raise ValueError(
                    f"Number of groups ({self.groups}) cannot be more than the number of channels ({self.channels})."
                )

            if self.channels % self.groups:
                raise ValueError(
                    f"Number of groups ({self.groups}) must be a multiple of the number of channels ({self.channels})."
                )

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"data_format": self.data_format, "groups": self._groups})

        del config["axis"]

        return config


@NORMALIZATIONS.register("frn")
@register_keras_serializable(package="SegMe>Policy>Normalization")
class FilterResponseNorm(layers.Layer):
    """Rewrite for any shape support"""

    def __init__(
        self,
        data_format=None,
        epsilon=1e-6,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        learned_epsilon=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

        self.data_format = backend.standardize_data_format(data_format)
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.use_eps_learned = learned_epsilon

    def build(self, input_shape):
        channels = (
            input_shape[-1]
            if "channels_last" == self.data_format
            else input_shape[1]
        )
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. Found `None`."
            )

        axis = list(range(1, len(input_shape)))
        if "channels_last" == self.data_format:
            self.axis = axis[:-1]
        else:
            self.axis = axis[1:]

        weight_shape = (
            [1, 1, 1, channels]
            if "channels_last" == self.data_format
            else [1, channels, 1, 1]
        )
        self.gamma = self.add_weight(
            shape=weight_shape,
            name="gamma",
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
        )
        self.beta = self.add_weight(
            shape=weight_shape,
            name="beta",
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
        )

        if self.use_eps_learned:
            self.eps_learned = self.add_weight(
                shape=(1,),
                name="learned_epsilon",
                initializer=initializers.Constant(1e-4),
                constraint=lambda e: tf.minimum(e, 0.0),
            )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, "float32")

        epsilon = self.epsilon
        if self.use_eps_learned:
            epsilon += tf.math.abs(self.eps_learned)

        nu2 = tf.reduce_mean(tf.square(outputs), axis=self.axis, keepdims=True)
        outputs = (
            outputs * tf.math.rsqrt(nu2 + epsilon) * self.gamma + self.beta
        )

        outputs = tf.cast(outputs, inputs.dtype)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "epsilon": self.epsilon,
                "learned_epsilon": self.use_eps_learned,
                "beta_initializer": initializers.serialize(
                    self.beta_initializer
                ),
                "gamma_initializer": initializers.serialize(
                    self.gamma_initializer
                ),
                "beta_regularizer": regularizers.serialize(
                    self.beta_regularizer
                ),
                "gamma_regularizer": regularizers.serialize(
                    self.gamma_regularizer
                ),
                "beta_constraint": constraints.serialize(self.beta_constraint),
                "gamma_constraint": constraints.serialize(
                    self.gamma_constraint
                ),
            }
        )

        return config
