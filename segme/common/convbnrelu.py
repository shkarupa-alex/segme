from tensorflow.keras import Sequential, activations, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>U2Net')
class ConvBnRelu(layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)

    @shape_type_conversion
    def build(self, input_shape):
        self.features = Sequential([
            layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding='same',
                dilation_rate=self.dilation_rate
            ),
            layers.BatchNormalization()])

        if 'linear' != self.activation:
            self.features.add(layers.Activation(self.activation))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.features(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation)
        })

        return config
