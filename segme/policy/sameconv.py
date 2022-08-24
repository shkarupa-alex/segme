from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.pad import SamePadding
from segme.policy.stdconv import StandardizedConv2D, StandardizedDepthwiseConv2D
from segme.policy.snconv import SpectralConv2D, SpectralDepthwiseConv2D
from segme.registry import LayerRegistry

SAMECONVS = LayerRegistry()


@SAMECONVS.register('conv')
@register_keras_serializable(package='SegMe>Policy>SameConv')
class SameConv(layers.Conv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), groups=1,
                 activation=None, use_bias=True, symmetric_pad=None, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        kwargs.pop('padding', None)
        self.same_pad = SamePadding(kernel_size, strides, dilation_rate, symmetric_pad=symmetric_pad)

        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=self.same_pad.padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        self.symmetric_pad = symmetric_pad
        self.inside_call = False

    @shape_type_conversion
    def build(self, input_shape):
        self.same_pad.build(input_shape)
        input_shape = self.same_pad.compute_output_shape(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        outputs = self.same_pad(inputs)

        self.inside_call = True
        outputs = super().call(outputs)
        self.inside_call = False

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.inside_call:
            output_shape = input_shape
        else:
            output_shape = self.same_pad.compute_output_shape(input_shape)

        return super().compute_output_shape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'symmetric_pad': self.symmetric_pad})
        del config['padding']

        return config


@SAMECONVS.register('stdconv')
@register_keras_serializable(package='SegMe>Policy>SameConv')
class SameStandardizedConv(StandardizedConv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), groups=1,
                 activation=None, use_bias=True, symmetric_pad=None, kernel_initializer=None, bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        kwargs.pop('padding', None)
        self.same_pad = SamePadding(kernel_size, strides, dilation_rate, symmetric_pad=symmetric_pad)

        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=self.same_pad.padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        self.symmetric_pad = symmetric_pad
        self.inside_call = False

    @shape_type_conversion
    def build(self, input_shape):
        self.same_pad.build(input_shape)
        input_shape = self.same_pad.compute_output_shape(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        outputs = self.same_pad(inputs)

        self.inside_call = True
        outputs = super().call(outputs)
        self.inside_call = False

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.inside_call:
            output_shape = input_shape
        else:
            output_shape = self.same_pad.compute_output_shape(input_shape)

        return super().compute_output_shape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'symmetric_pad': self.symmetric_pad})
        del config['padding']

        return config


@SAMECONVS.register('snconv')
@register_keras_serializable(package='SegMe>Policy>SameConv')
class SameSpectralConv(SpectralConv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), groups=1,
                 activation=None, use_bias=True, symmetric_pad=None, kernel_initializer=None, bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        kwargs.pop('padding', None)
        self.same_pad = SamePadding(kernel_size, strides, dilation_rate, symmetric_pad=symmetric_pad)

        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=self.same_pad.padding,
                         data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)

        self.symmetric_pad = symmetric_pad
        self.inside_call = False

    @shape_type_conversion
    def build(self, input_shape):
        self.same_pad.build(input_shape)
        input_shape = self.same_pad.compute_output_shape(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        outputs = self.same_pad(inputs)

        self.inside_call = True
        outputs = super().call(outputs)
        self.inside_call = False

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.inside_call:
            output_shape = input_shape
        else:
            output_shape = self.same_pad.compute_output_shape(input_shape)

        return super().compute_output_shape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'symmetric_pad': self.symmetric_pad})
        del config['padding']

        return config


@SAMECONVS.register('dwconv')
@register_keras_serializable(package='SegMe>Policy>SameConv')
class SameDepthwiseConv(layers.DepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), depth_multiplier=1, data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, symmetric_pad=None, kernel_initializer=None,
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        kwargs.pop('padding', None)
        self.same_pad = SamePadding(kernel_size, strides, dilation_rate, symmetric_pad=symmetric_pad)

        super().__init__(kernel_size=kernel_size, strides=strides, padding=self.same_pad.padding,
                         depth_multiplier=depth_multiplier, data_format=data_format, dilation_rate=dilation_rate,
                         activation=activation, use_bias=use_bias, depthwise_initializer=kernel_initializer,
                         bias_initializer=bias_initializer, depthwise_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                         depthwise_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)

        self.symmetric_pad = symmetric_pad

    @shape_type_conversion
    def build(self, input_shape):
        self.same_pad.build(input_shape)
        output_shape = self.same_pad.compute_output_shape(input_shape)

        super().build(output_shape)

    def call(self, inputs):
        outputs = self.same_pad(inputs)

        return super().call(outputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.same_pad.compute_output_shape(input_shape)

        return super().compute_output_shape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_initializer': config['depthwise_initializer'],
            'kernel_regularizer': config['depthwise_regularizer'],
            'kernel_constraint': config['depthwise_constraint'],
            'symmetric_pad': self.symmetric_pad
        })

        del config['depthwise_initializer']
        del config['depthwise_regularizer']
        del config['depthwise_constraint']
        del config['padding']

        return config


@SAMECONVS.register('dwstdconv')
@register_keras_serializable(package='SegMe>Policy>SameConv')
class SameStandardizedDepthwiseConv(StandardizedDepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), depth_multiplier=1, data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, symmetric_pad=None, kernel_initializer=None,
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        kwargs.pop('padding', None)
        self.same_pad = SamePadding(kernel_size, strides, dilation_rate, symmetric_pad=symmetric_pad)

        super().__init__(kernel_size=kernel_size, strides=strides, padding=self.same_pad.padding,
                         depth_multiplier=depth_multiplier, data_format=data_format, dilation_rate=dilation_rate,
                         activation=activation, use_bias=use_bias, depthwise_initializer=kernel_initializer,
                         bias_initializer=bias_initializer, depthwise_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                         depthwise_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)

        self.symmetric_pad = symmetric_pad

    @shape_type_conversion
    def build(self, input_shape):
        self.same_pad.build(input_shape)
        output_shape = self.same_pad.compute_output_shape(input_shape)

        super().build(output_shape)

    def call(self, inputs):
        outputs = self.same_pad(inputs)

        return super().call(outputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.same_pad.compute_output_shape(input_shape)

        return super().compute_output_shape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_initializer': config['depthwise_initializer'],
            'kernel_regularizer': config['depthwise_regularizer'],
            'kernel_constraint': config['depthwise_constraint'],
            'symmetric_pad': self.symmetric_pad
        })

        del config['depthwise_initializer']
        del config['depthwise_regularizer']
        del config['depthwise_constraint']
        del config['padding']

        return config


@SAMECONVS.register('dwsnconv')
@register_keras_serializable(package='SegMe>Policy>SameConv')
class SameSpectralDepthwiseConv(SpectralDepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), depth_multiplier=1, data_format=None, dilation_rate=(1, 1),
                 activation=None, use_bias=True, symmetric_pad=None, kernel_initializer=None,
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        kwargs.pop('padding', None)
        self.same_pad = SamePadding(kernel_size, strides, dilation_rate, symmetric_pad=symmetric_pad)

        super().__init__(kernel_size=kernel_size, strides=strides, padding=self.same_pad.padding,
                         depth_multiplier=depth_multiplier, data_format=data_format, dilation_rate=dilation_rate,
                         activation=activation, use_bias=use_bias, depthwise_initializer=kernel_initializer,
                         bias_initializer=bias_initializer, depthwise_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                         depthwise_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)

        self.symmetric_pad = symmetric_pad

    @shape_type_conversion
    def build(self, input_shape):
        self.same_pad.build(input_shape)
        output_shape = self.same_pad.compute_output_shape(input_shape)

        super().build(output_shape)

    def call(self, inputs):
        outputs = self.same_pad(inputs)

        return super().call(outputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.same_pad.compute_output_shape(input_shape)

        return super().compute_output_shape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_initializer': config['depthwise_initializer'],
            'kernel_regularizer': config['depthwise_regularizer'],
            'kernel_constraint': config['depthwise_constraint'],
            'symmetric_pad': self.symmetric_pad
        })

        del config['depthwise_initializer']
        del config['depthwise_regularizer']
        del config['depthwise_constraint']
        del config['padding']

        return config
