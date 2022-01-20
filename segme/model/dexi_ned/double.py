from keras import Sequential
from keras import activations, layers, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu


@register_keras_serializable(package='SegMe>DexiNed')
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
        self.features = Sequential([
            ConvNormRelu(self.mid_features, 3, padding='same', strides=self.stride,
                         kernel_regularizer=regularizers.l2(1e-3)),
            ConvNormRelu(self._out_features, 3, padding='same', activation=self.activation,
                         kernel_regularizer=regularizers.l2(1e-3)),
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
