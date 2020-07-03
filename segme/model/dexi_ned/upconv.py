import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>DexiNed')
class UpConvBlock(layers.Layer):
    def __init__(self, up_scale, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.up_scale = up_scale
        self._constant_features = 16

    @shape_type_conversion
    def build(self, input_shape):
        total_up_scale = 2 ** self.up_scale
        trunc_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)

        self.features = Sequential()
        for i in range(self.up_scale):
            is_last = i == self.up_scale - 1
            out_features = 1 if is_last else self._constant_features
            kernel_init = trunc_init if is_last else 'glorot_uniform'

            self.features.add(layers.Conv2D(
                filters=out_features,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='relu',
                kernel_initializer=kernel_init))
            self.features.add(layers.Conv2DTranspose(
                out_features,
                kernel_size=(total_up_scale, total_up_scale),
                strides=2,
                padding='same',
                kernel_initializer=kernel_init))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.features(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.features.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'up_scale': self.up_scale})

        return config
