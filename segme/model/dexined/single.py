import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, regularizers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class DexiNedSingleConvBlock(layers.Layer):
    def __init__(
            self, out_features, kernel_size=1, stride=1, weight_norm=False,
            kernel_initializer='glorot_uniform', kernel_l2=None,
            bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_norm = weight_norm
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_l2 = kernel_l2
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    @shape_type_conversion
    def build(self, input_shape):
        kernel_regularizer = None
        if self.kernel_l2 is not None:
            kernel_regularizer = regularizers.l2(self.kernel_l2)

        self.features = Sequential([layers.Conv2D(
            filters=self.out_features,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=self.bias_initializer)])
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
            'weight_norm': self.weight_norm,
            'kernel_initializer': tf.keras.initializers.serialize(
                self.kernel_initializer),
            'kernel_l2': self.kernel_l2,
            'bias_initializer': tf.keras.initializers.serialize(
                self.bias_initializer)
        })

        return config
