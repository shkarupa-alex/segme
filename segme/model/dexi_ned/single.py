import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import initializers, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>DexiNed')
class SingleConvBlock(layers.Layer):
    def __init__(self, out_features, kernel_size=1, stride=1, weight_norm=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_norm = weight_norm

    @shape_type_conversion
    def build(self, input_shape):
        kernel_init = initializers.random_normal(stddev=0.01)
        self.features = Sequential([layers.Conv2D(
            filters=self.out_features,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            kernel_initializer=kernel_init)])
        if self.weight_norm:
            self.features.add(layers.BatchNormalization())

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.features(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.features.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_features': self.out_features,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'weight_norm': self.weight_norm
        })

        return config
