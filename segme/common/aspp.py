import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.internear import NearestInterpolation
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Common')
class AtrousSpatialPyramidPooling(layers.Layer):
    _stride_rates = {
        8: [12, 24, 36],
        16: [6, 12, 18],
        32: [3, 6, 9]
    }

    def __init__(self, filters, stride, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.stride = stride
        self.dropout = dropout

        if stride not in self._stride_rates:
            raise NotImplementedError('Unsupported input stride')

    @shape_type_conversion
    def build(self, input_shape):
        self.conv1 = ConvNormAct(self.filters, 1)

        rate0, rate1, rate2 = self._stride_rates[self.stride]
        self.conv3r0 = Sequential([
            ConvNormAct(None, 3, conv_kwargs={'dilation_rate': rate0, 'symmetric_pad': False}),
            ConvNormAct(self.filters, 1)])
        self.conv3r1 = Sequential([
            ConvNormAct(None, 3, conv_kwargs={'dilation_rate': rate1, 'symmetric_pad': False}),
            ConvNormAct(self.filters, 1)])
        self.conv3r2 = Sequential([
            ConvNormAct(None, 3, conv_kwargs={'dilation_rate': rate2, 'symmetric_pad': False}),
            ConvNormAct(self.filters, 1)])

        self.pool = Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            # TODO: wait for https://github.com/tensorflow/tensorflow/issues/48845
            ConvNormAct(self.filters, 1, norm_kwargs={'fused': False})
        ])
        self.intnear = NearestInterpolation()

        self.proj = Sequential([
            ConvNormAct(self.filters, 1),
            layers.Dropout(self.dropout)
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.concat([
            self.conv1(inputs),
            self.conv3r0(inputs),
            self.conv3r1(inputs),
            self.conv3r2(inputs),
            self.intnear([self.pool(inputs), inputs])
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
            'dropout': self.dropout
        })

        return config
