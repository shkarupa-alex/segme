import tensorflow as tf
from keras import activations, constraints, initializers, layers, models, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .convnormrelu import ConvNormRelu, DepthwiseConvNormRelu
from .resizebysample import resize_by_sample


@register_keras_serializable(package='SegMe')
class AtrousSeparableConv(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), depth_multiplier=1, data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform',
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
                 kernel_constraint=None, bias_constraint=None, standardized=False, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.depthwise = DepthwiseConvNormRelu(
            kernel_size=self.kernel_size, strides=self.strides, depth_multiplier=self.depth_multiplier,
            data_format=self.data_format, dilation_rate=self.dilation_rate, activation=self.activation,
            depthwise_initializer=self.depthwise_initializer, bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.depthwise_regularizer, bias_regularizer=self.bias_regularizer,
            depthwise_constraint=self.depthwise_constraint, bias_constraint=self.bias_constraint,
            standardized=self.standardized)
        self.pointwise = ConvNormRelu(
            self.filters, 1, strides=self.strides, data_format=self.data_format, dilation_rate=1,
            groups=self.groups, activation=self.activation, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint, standardized=self.standardized)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.depthwise(inputs)
        outputs = self.pointwise(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'depth_multiplier': self.depth_multiplier,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'depthwise_initializer': initializers.serialize(self.depthwise_initializer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'depthwise_regularizer': regularizers.serialize(self.depthwise_regularizer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'depthwise_constraint': constraints.serialize(self.depthwise_constraint),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'standardized': self.standardized
        })

        return config


@register_keras_serializable(package='SegMe')
class AtrousSpatialPyramidPooling(layers.Layer):
    _stride_rates = {
        8: [12, 24, 36],
        16: [6, 12, 18],
        32: [3, 6, 9]
    }

    def __init__(self, filters, stride, dropout=0.1, activation='relu', standardized=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.stride = stride
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.standardized = standardized

        if stride not in self._stride_rates:
            raise NotImplementedError('Unsupported input stride')

    @shape_type_conversion
    def build(self, input_shape):
        self.conv1 = ConvNormRelu(self.filters, 1, activation=self.activation, standardized=self.standardized)

        rate0, rate1, rate2 = self._stride_rates[self.stride]
        self.conv3r0 = AtrousSeparableConv(
            self.filters, 3, dilation_rate=rate0, activation=self.activation, standardized=self.standardized)
        self.conv3r1 = AtrousSeparableConv(
            self.filters, 3, dilation_rate=rate1, activation=self.activation, standardized=self.standardized)
        self.conv3r2 = AtrousSeparableConv(
            self.filters, 3, dilation_rate=rate2, activation=self.activation, standardized=self.standardized)

        self.pool = models.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            # TODO: wait for https://github.com/tensorflow/tensorflow/issues/48845
            ConvNormRelu(self.filters, 1, activation=self.activation, standardized=self.standardized, bn_fused=False)
        ])

        self.proj = models.Sequential([
            ConvNormRelu(self.filters, 1, activation=self.activation, standardized=self.standardized),
            layers.Dropout(self.dropout)  # 0.5 in some implementations
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.concat([
            self.conv1(inputs),
            self.conv3r0(inputs),
            self.conv3r1(inputs),
            self.conv3r2(inputs),
            resize_by_sample([self.pool(inputs), inputs])
        ], axis=-1)
        outputs = self.proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'stride': self.stride,
            'dropout': self.dropout,
            'activation': activations.serialize(self.activation),
            'standardized': self.standardized
        })

        return config
