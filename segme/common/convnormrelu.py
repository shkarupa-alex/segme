from keras import activations, constraints, initializers, layers, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.layers import GroupNormalization
from .stdconv import StandardizedConv2D


@register_keras_serializable(package='SegMe')
class ConvNormRelu(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', dilation_rate=(1, 1), groups=1,
                 activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, standardized=False, bn_fused=True, gn_groups=32, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.standardized = standardized
        self.bn_fused = bn_fused
        self.gn_groups = gn_groups

    def _norm(self, **kwargs):
        return layers.BatchNormalization(fused=self.norm_fused, **kwargs)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        conv_kwargs = {
            'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides, 'padding': self.padding,
            'dilation_rate': self.dilation_rate, 'groups': self.groups, 'use_bias': False,
            'kernel_initializer': self.kernel_initializer, 'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer, 'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint, 'bias_constraint': self.bias_constraint}
        conv_layer = StandardizedConv2D if self.standardized else layers.Conv2D
        self.conv = conv_layer(**conv_kwargs)

        norm_kwargs = {'groups': self.gn_groups} if self.standardized else {'fused': self.bn_fused}
        norm_layer = GroupNormalization if self.standardized else layers.BatchNormalization
        self.norm = norm_layer(**norm_kwargs)

        self.act = None
        if self.activation not in {None, 'linear'}:
            self.act = layers.Activation(self.activation)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)

        if self.act is not None:
            outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'standardized': self.standardized,
            'bn_fused': self.bn_fused,
            'gn_groups': self.gn_groups
        })

        return config
