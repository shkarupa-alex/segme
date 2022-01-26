from keras import activations, constraints, initializers, layers, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .stdconv import StandardizedConv2D, StandardizedDepthwiseConv2D


@register_keras_serializable(package='SegMe')
class SameConv(layers.Conv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), groups=1,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        if 'padding' in kwargs:
            raise ValueError('SameConv layer supports only "same" padding')
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

    def get_config(self):
        config = super().get_config()
        del config['padding']

        return config


@register_keras_serializable(package='SegMe')
class SameDepthwiseConv(layers.DepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), depth_multiplier=1, data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros',
                 depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 depthwise_constraint=None, bias_constraint=None, **kwargs):
        if 'padding' in kwargs:
            raise ValueError('SameDepthwiseConv layer supports only "same" padding')
        super().__init__(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=depth_multiplier,
                         data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                         depthwise_initializer=depthwise_initializer, bias_initializer=bias_initializer,
                         depthwise_regularizer=depthwise_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, depthwise_constraint=depthwise_constraint,
                         bias_constraint=bias_constraint, **kwargs)

    def get_config(self):
        config = super().get_config()
        del config['padding']

        return config


@register_keras_serializable(package='SegMe')
class SameStandardizedConv(StandardizedConv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), groups=1,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        if 'padding' in kwargs:
            raise ValueError('SameStandardizedConv layer supports only "same" padding')
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

    def get_config(self):
        config = super().get_config()
        del config['padding']

        return config


@register_keras_serializable(package='SegMe')
class SameStandardizedDepthwiseConv(StandardizedDepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), depth_multiplier=1, data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros',
                 depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 depthwise_constraint=None, bias_constraint=None, **kwargs):
        if 'padding' in kwargs:
            raise ValueError('SameDepthwiseConv layer supports only "same" padding')
        super().__init__(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=depth_multiplier,
                         data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                         depthwise_initializer=depthwise_initializer, bias_initializer=bias_initializer,
                         depthwise_regularizer=depthwise_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, depthwise_constraint=depthwise_constraint,
                         bias_constraint=bias_constraint, **kwargs)

    def get_config(self):
        config = super().get_config()
        del config['padding']

        return config
