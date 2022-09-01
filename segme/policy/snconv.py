import tensorflow as tf
from keras import backend, initializers, layers
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Policy')
class SpectralConv2D(layers.Conv2D):
    """Implements https://arxiv.org/abs/1802.05957"""

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, use_bias=True, power_iterations=1, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        self.power_iterations = power_iterations

    @shape_type_conversion
    def build(self, input_shape):
        self.u = self.add_weight(
            shape=(1, self.filters),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=self.dtype,
        )

        super().build(input_shape)

    def normalize_call(self, inputs):
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

        with tf.control_dependencies([self.kernel.assign(kernel), self.u.assign(u)]):
            outputs = tf.identity(inputs)

        return outputs

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(
            training,
            lambda: self.normalize_call(inputs),
            lambda: tf.identity(inputs))

        outputs = super().call(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'power_iterations': self.power_iterations})

        return config
