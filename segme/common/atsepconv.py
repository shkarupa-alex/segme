from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class AtrousSepConv(layers.Layer):
    def __init__(self, filters, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters
        self.dilation = dilation

    @shape_type_conversion
    def build(self, input_shape):
        self.conv = Sequential([
            layers.DepthwiseConv2D(3, padding='same', dilation_rate=self.dilation, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.filters, 1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'dilation': self.dilation
        })

        return config
