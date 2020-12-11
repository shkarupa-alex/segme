from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>PyramidFeatureAttention')
class CFE(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        rates = [3, 5, 7]
        self.cfe0 = layers.Conv2D(self.filters, 1, padding='same', use_bias=False, name='cfe0')
        self.cfe1 = layers.Conv2D(self.filters, 3, dilation_rate=rates[0], padding='same', use_bias=False, name='cfe1')
        self.cfe2 = layers.Conv2D(self.filters, 3, dilation_rate=rates[1], padding='same', use_bias=False, name='cfe2')
        self.cfe3 = layers.Conv2D(self.filters, 3, dilation_rate=rates[2], padding='same', use_bias=False, name='cfe3')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = layers.concatenate([
            self.cfe0(inputs),
            self.cfe1(inputs),
            self.cfe2(inputs),
            self.cfe3(inputs),
        ])
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters * 4,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
