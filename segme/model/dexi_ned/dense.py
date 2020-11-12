from tensorflow.keras import Sequential
from tensorflow.keras import layers, utils, regularizers
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>DexiNed')
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
                layers.Conv2D(
                    filters=self.out_features,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    kernel_regularizer=regularizers.l2(1e-3)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    filters=self.out_features,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    kernel_regularizer=regularizers.l2(1e-3)),
                layers.BatchNormalization(),
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
