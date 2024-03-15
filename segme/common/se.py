from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvAct
from segme.common.sequence import Sequence


@register_keras_serializable(package='SegMe>Common')
class SE(layers.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if not 0. <= ratio <= 1.:
            raise ValueError('Squeeze ratio must be in range [0; 1].')

        self.ratio = ratio

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        filters = max(1, int(channels * self.ratio))
        self.se = Sequence([
            layers.GlobalAvgPool2D(keepdims=True, name='pool'),
            ConvAct(filters, 1, kernel_initializer='variance_scaling', name='fc0'),
            layers.Conv2D(channels, 1, activation='sigmoid', kernel_initializer='variance_scaling', name='fc1')
        ], name='se')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs * self.se(inputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})

        return config
