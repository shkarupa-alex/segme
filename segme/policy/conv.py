import tensorflow as tf
from keras import backend, initializers, layers
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy.registry import LayerRegistry

CONVOLUTIONS = LayerRegistry()


@CONVOLUTIONS.register('conv')
@register_keras_serializable(package='SegMe>Policy>Conv')
class FixedConv(layers.Conv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=1, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

    @shape_type_conversion
    def build(self, input_shape):
        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError('Strides > 1 not supported in conjunction with dilations')

        super().build(input_shape)

    def convolution_op(self, inputs, kernel):
        paddings = 'VALID' if 'same' != self.padding else 'SAME'

        if 'SAME' == paddings and max(self.kernel_size) > 1 and max(self.strides) > 1:
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)
            paddings = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
            paddings = ((0, 0),) + paddings if self.data_format == 'channels_first' else paddings + ((0, 0),)

            return tf.nn.conv2d(
                inputs, kernel, strides=list(self.strides), padding=list(paddings), dilations=list(self.dilation_rate),
                data_format=self._tf_data_format, name=self.__class__.__name__)

        return tf.nn.convolution(
            inputs, kernel, strides=list(self.strides), padding=paddings, dilations=list(self.dilation_rate),
            data_format=self._tf_data_format, name=self.__class__.__name__)

    def get_config(self):
        config = super().get_config()
        del config['groups']

        return config


@CONVOLUTIONS.register('dwconv')
@register_keras_serializable(package='SegMe>Policy>Conv')
class FixedDepthwiseConv(layers.DepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        kwargs.pop('depth_multiplier', None)
        super().__init__(kernel_size=kernel_size, strides=strides, padding=padding, depth_multiplier=1,
                         data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                         depthwise_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         depthwise_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, depthwise_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

    @shape_type_conversion
    def build(self, input_shape):
        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError('Strides > 1 not supported in conjunction with dilations')

        super().build(input_shape)

    def _conv_op(self, inputs, kernel):
        strides = (1, 1) + self.strides if self.data_format == 'channels_first' else (1,) + self.strides + (1,)
        paddings = 'VALID' if 'same' != self.padding else 'SAME'

        if 'SAME' == paddings and max(self.kernel_size) > 1 and max(self.strides) > 1:
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)
            paddings = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
            paddings = ((0, 0),) + paddings if self.data_format == 'channels_first' else paddings + ((0, 0),)

        return tf.nn.depthwise_conv2d(
            inputs, kernel, strides=strides, padding=paddings, dilations=self.dilation_rate,
            data_format=self._tf_data_format)

    def call(self, inputs):
        outputs = self._conv_op(inputs, self.depthwise_kernel)

        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_initializer': config['depthwise_initializer'],
            'kernel_regularizer': config['depthwise_regularizer'],
            'kernel_constraint': config['depthwise_constraint']
        })

        del config['depth_multiplier']
        del config['depthwise_initializer']
        del config['depthwise_regularizer']
        del config['depthwise_constraint']

        return config


@CONVOLUTIONS.register('stdconv')
@register_keras_serializable(package='SegMe>Policy>Conv')
class StandardizedConv(FixedConv):
    """Implements https://arxiv.org/abs/1903.10520"""

    def before_train(self, inputs):
        kernel = tf.cast(self.kernel, self.dtype)

        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        kernel = tf.nn.batch_normalization(kernel, mean, var, None, None, variance_epsilon=1e-5)
        kernel = tf.stop_gradient(kernel)

        with tf.control_dependencies([self.kernel.assign(kernel)]):
            outputs = tf.identity(inputs)

        return outputs

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(
            training,
            lambda: self.before_train(inputs),
            lambda: tf.identity(inputs))

        outputs = super().call(outputs)

        return outputs


@CONVOLUTIONS.register('snconv')
@register_keras_serializable(package='SegMe>Policy>Conv')
class SpectralConv(StandardizedConv):
    """Implements https://arxiv.org/abs/1802.05957"""

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, power_iterations=1, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
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

        with tf.control_dependencies([self.kernel.assign(kernel), self.u.assign(u)]):
            outputs = tf.identity(inputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'power_iterations': self.power_iterations})

        return config
