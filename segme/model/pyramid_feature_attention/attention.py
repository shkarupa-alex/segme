import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>PyramidFeatureAttention')
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if self.channels % 2:
            raise ValueError('Channel dimension of the inputs should be divided by 2 without a remainder.')

        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        k = 9
        self.cbr0 = Sequential([
            layers.Conv2D(self.channels // 2, (1, k), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, (k, 1), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.cbr1 = Sequential([
            layers.Conv2D(self.channels // 2, (k, 1), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, (1, k), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.sigmoid = layers.Activation('sigmoid')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = layers.add([
            self.cbr0(inputs),
            self.cbr1(inputs)
        ])
        outputs = self.sigmoid(outputs)
        outputs = tf.tile(outputs, [1, 1, 1, self.channels])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


@utils.register_keras_serializable(package='SegMe>PyramidFeatureAttention')
class ChannelWiseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if channels % 4:
            raise ValueError('Channel dimension of the inputs should be divided by 4 without a remainder.')

        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.pool = layers.GlobalAveragePooling2D()
        self.dense0 = layers.Dense(channels // 4, activation='relu')
        self.dense1 = layers.Dense(channels, activation='sigmoid', activity_regularizer=self.mean_regularizer)
        self.reshape = layers.Reshape([1, 1, channels])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_shape = tf.shape(inputs)

        outputs = self.pool(inputs)
        outputs = self.dense0(outputs)
        outputs = self.dense1(outputs)

        outputs = self.reshape(outputs)
        outputs = tf.tile(outputs, [1, inputs_shape[1], inputs_shape[2], 1])
        outputs = layers.multiply([outputs, inputs])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def mean_regularizer(self, x):
        return tf.reduce_mean(x)
