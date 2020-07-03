from tensorflow.keras import Sequential
from tensorflow.keras import activations, initializers, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>DexiNed')
class DoubleConvBlock(layers.Layer):
    def __init__(self, mid_features, out_features=None, stride=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.mid_features = mid_features
        self.out_features = out_features
        self._out_features = self.out_features or self.mid_features
        self.stride = stride
        self.activation = activations.get(activation)

    @shape_type_conversion
    def build(self, input_shape):
        kernel_init = initializers.random_normal(stddev=0.01)
        self.features = Sequential([
            layers.Conv2D(
                filters=self.mid_features,
                kernel_size=3,
                strides=self.stride,
                padding='same',
                kernel_initializer=kernel_init),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(
                filters=self._out_features,
                kernel_size=3,
                padding='same',
                strides=1,
                kernel_initializer=kernel_init),
            layers.BatchNormalization(),
            layers.Activation(self.activation)
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.features(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.features.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mid_features': self.mid_features,
            'out_features': self.out_features,
            'stride': self.stride,
            'activation': activations.serialize(self.activation)
        })

        return config
