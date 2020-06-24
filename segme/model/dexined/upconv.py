import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, regularizers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class DexiNedUpConvBlock(layers.Layer):
    def __init__(
            self, up_scale, kernel_initializer='glorot_uniform',
            kernel_l2=None, bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.up_scale = up_scale
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_l2 = kernel_l2
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._constant_features = 16

    @shape_type_conversion
    def build(self, input_shape):
        kernel_regularizer = None
        if self.kernel_l2 is not None:
            kernel_regularizer = regularizers.l2(self.kernel_l2)

        total_up_scale = 2 ** self.up_scale
        trunc_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)

        self.features = Sequential()
        for i in range(self.up_scale):
            is_last = i == self.up_scale - 1
            out_features = 1 if is_last else self._constant_features
            kernel_init = trunc_init if is_last else self.kernel_initializer
            bias_init = self.bias_initializer if is_last else 'zeros'

            self.features.add(layers.Conv2D(
                filters=out_features,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='relu',
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_regularizer))
            self.features.add(layers.Conv2DTranspose(
                out_features,
                kernel_size=(total_up_scale, total_up_scale),
                strides=2,
                padding='same',
                kernel_initializer=kernel_init,
                kernel_regularizer=kernel_regularizer,
                bias_initializer=bias_init))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.features(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.features.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'up_scale': self.up_scale,
            'kernel_initializer': tf.keras.initializers.serialize(
                self.kernel_initializer),
            'kernel_l2': self.kernel_l2,
            'bias_initializer': tf.keras.initializers.serialize(
                self.bias_initializer)
        })

        return config
