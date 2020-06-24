import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, regularizers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class DexiNedDenseBlock(layers.Layer):
    def __init__(
            self, num_layers, out_features, kernel_initializer='glorot_uniform',
            kernel_l2=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features
            layers.InputSpec(ndim=4)  # skip
        ]
        self.num_layers = num_layers
        self.out_features = out_features  # TODO: estimate from input shape
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_l2 = kernel_l2

    @shape_type_conversion
    def build(self, input_shape):
        kernel_regularizer = None
        if self.kernel_l2 is not None:
            kernel_regularizer = regularizers.l2(self.kernel_l2)

        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(Sequential([
                layers.ReLU(),
                layers.Conv2D(
                    filters=self.out_features,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    filters=self.out_features,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                layers.BatchNormalization(),
            ]))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        features, skip = inputs

        for layer in self.layers:
            features = layer(features)
            features = 0.5 * layers.add([features, skip])

        return features, skip

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        features_shape, skip_shape = input_shape

        return features_shape[:-1] + (self.out_features,), skip_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'out_features': self.out_features,
            'kernel_initializer': tf.keras.initializers.serialize(
                self.kernel_initializer),
            'kernel_l2': self.kernel_l2
        })

        return config
