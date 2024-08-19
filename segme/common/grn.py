import tensorflow as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import layers
from keras.src import regularizers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class GRN(layers.Layer):
    """Inspired with : https://arxiv.org/pdf/2301.00808.pdf"""

    def __init__(
        self,
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
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.supports_masking = True

        if "float32" != self.dtype:
            raise ValueError(
                f"Global response normalization is only supported when the "
                f"variables dtype is float32. Got dtype: {self.dtype}."
            )

        if epsilon < 1.001e-5:
            raise ValueError(
                f"Global response normalization is not supported for "
                f"epsilon {epsilon} (<1.001e-5)."
            )

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        if self.scale:
            self.gamma = self.add_weight(
                shape=[1, 1, 1, channels],
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        if self.center:
            self.beta = self.add_weight(
                shape=[1, 1, 1, channels],
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        # Accumulating sum(x^2) in spatially large feature maps leads to
        # Inf values
        # gx = tf.norm(inputs, axis=[1, 2], keepdims=True)

        # Here we will use response normalization method from
        # https://arxiv.org/pdf/1911.09737.pdf
        gx = tf.reduce_mean(tf.square(inputs), axis=[1, 2], keepdims=True)

        # Simple divisive normalization works best,
        nx = gx / (tf.reduce_mean(gx, axis=-1, keepdims=True) + self.epsilon)

        # though standardization (||Xi|| − µ)/σ yields similar results.
        # batch = tf.shape(inputs)[0]
        # scale = tf.ones([batch], dtype=self.dtype)
        # offset = tf.zeros([batch], dtype=self.dtype)
        # nx = tf.squeeze(gx, axis=1)[None]
        # nx, _, _ = tf.compat.v1.nn.fused_batch_norm(
        #     nx, scale=scale, offset=offset, epsilon=self.epsilon,
        #     data_format='NCHW')
        # nx = tf.squeeze(nx, axis=0)[:, None]

        if self.scale:
            nx *= self.gamma

        outputs = inputs * (nx + 1.0)

        if self.center:
            outputs += self.beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
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
