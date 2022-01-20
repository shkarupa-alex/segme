from keras import Sequential
from keras import layers, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu


@register_keras_serializable(package='SegMe>DexiNed')
class DenseBlock(layers.Layer):
    def __init__(self, num_layers, out_features, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features
            layers.InputSpec(ndim=4)  # skip
        ]
        self.num_layers = num_layers
        self.out_features = out_features

    @shape_type_conversion
    def build(self, input_shape):
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(Sequential([
                layers.ReLU(),
                ConvNormRelu(self.out_features, 3, padding='same', kernel_regularizer=regularizers.l2(1e-3)),
                ConvNormRelu(self.out_features, 3, padding='same', activation='linear', kernel_regularizer=regularizers.l2(1e-3)),
            ]))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        features, skip = inputs

        for layer in self.layers:
            features = layer(features)
            features = layers.add([features, skip]) / 2.

        return features

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        features_shape, skip_shape = input_shape

        return features_shape[:-1] + (self.out_features,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'out_features': self.out_features,
        })

        return config
