from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Common')
class SE(layers.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.ratio = ratio

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        filters = max(1, int(channels * self.ratio))
        self.se = Sequential([
            layers.GlobalAvgPool2D(keepdims=True),
            ConvNormAct(filters, 1, conv_kwargs={'kernel_initializer': 'variance_scaling'}, norm_type=None),
            layers.Conv2D(channels, 1, activation='sigmoid', kernel_initializer='variance_scaling')
        ])

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
