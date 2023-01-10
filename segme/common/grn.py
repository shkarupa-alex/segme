import tensorflow as tf
from keras import backend, constraints, initializers, layers, regularizers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class GRN(layers.Layer):
    """ Proposed in: https://arxiv.org/pdf/2301.00808.pdf """

    def __init__(self, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.supports_masking = True

        if 'float32' != self.dtype:
            raise ValueError(
                f'Global response normalization is only supported when the variables dtype is '
                f'float32. Got dtype: {self.dtype}.')

        if self._compute_dtype not in ('float16', 'float16', 'float32', None):
            raise ValueError(
                f'Global response normalization is only supported when the compute dtype is '
                f'float16, bfloat16, or float32. Got dtype: {self._compute_dtype}.')

        if epsilon < 1.001e-5:
            raise ValueError(
                f'Global response normalization is not supported for epsilon {epsilon} (<1.001e-5).')

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        if self.scale:
            self.gamma = self.add_weight(
                shape=[1, 1, 1, channels], name='gamma', initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)
        if self.center:
            self.beta = self.add_weight(
                shape=[1, 1, 1, channels], name='beta', initializer=self.beta_initializer,
                regularizer=self.beta_regularizer, constraint=self.beta_constraint)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        gx = tf.norm(inputs, axis=[1, 2], keepdims=True)

        # Simple divisive normalization works best,
        # nx = gx / (tf.reduce_mean(gx, axis=-1, keepdims=True) + self.epsilon)
        # though standardization (||Xi|| − µ)/σ yields similar results.

        # Here we will use standardization due to faster/stabler fused implementation.
        batch = tf.shape(inputs)[0]
        scale = tf.ones([batch], dtype=self.dtype)
        offset = tf.zeros([batch], dtype=self.dtype)
        nx = tf.squeeze(gx, axis=1)[None]
        nx, _, _ = tf.compat.v1.nn.fused_batch_norm(
            nx, scale=scale, offset=offset, epsilon=self.epsilon, data_format='NCHW')
        nx = tf.squeeze(nx, axis=0)[:, None]

        if self.scale:
            nx *= self.gamma

        outputs = inputs * (nx + 1.)

        if self.center:
            outputs += self.beta

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        })

        return config
