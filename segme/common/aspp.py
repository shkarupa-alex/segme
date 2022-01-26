import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .convnormrelu import ConvNormRelu, DepthwiseConvNormRelu
from .resizebysample import resize_by_sample


@register_keras_serializable(package='SegMe')
class AtrousSeparableConv(layers.Layer):
    def __init__(self, filters, dilation, standardized, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.dilation = dilation
        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.depthwise = DepthwiseConvNormRelu(3, dilation_rate=self.dilation, standardized=self.standardized)
        self.pointwise = ConvNormRelu(self.filters, 1, standardized=self.standardized)

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
            'dilation': self.dilation,
            'standardized': self.standardized
        })

        return config


@register_keras_serializable(package='SegMe')
class ASPPPool(layers.Layer):
    def __init__(self, filters, standardized, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.pool = models.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            # TODO: wait for https://github.com/tensorflow/tensorflow/issues/48845
            ConvNormRelu(self.filters, 1, standardized=self.standardized, bn_fused=False)
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.pool(inputs)
        outputs = resize_by_sample([outputs, inputs])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'standardized': self.standardized
        })

        return config


@register_keras_serializable(package='SegMe')
class ASPP(layers.Layer):
    _stride_rates = {
        8: [12, 24, 36],
        16: [6, 12, 18],
        32: [3, 6, 9]
    }

    def __init__(self, filters, stride, dropout=0.1, standardized=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.stride = stride
        self.dropout = dropout
        self.standardized = standardized

        if stride not in self._stride_rates:
            raise NotImplementedError('Unsupported input stride')

    @shape_type_conversion
    def build(self, input_shape):
        self.conv1 = ConvNormRelu(self.filters, 1, standardized=self.standardized)

        rate0, rate1, rate2 = self._stride_rates[self.stride]
        self.conv3r0 = AtrousSeparableConv(filters=self.filters, dilation=rate0, standardized=self.standardized)
        self.conv3r1 = AtrousSeparableConv(filters=self.filters, dilation=rate1, standardized=self.standardized)
        self.conv3r2 = AtrousSeparableConv(filters=self.filters, dilation=rate2, standardized=self.standardized)

        self.pool = ASPPPool(filters=self.filters, standardized=self.standardized)

        self.proj = models.Sequential([
            ConvNormRelu(self.filters, 1, standardized=self.standardized),
            layers.Dropout(self.dropout)  # 0.5 in some implementations
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.concat([
            self.conv1(inputs),
            self.conv3r0(inputs),
            self.conv3r1(inputs),
            self.conv3r2(inputs),
            self.pool(inputs)
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
            'standardized': self.standardized
        })

        return config
