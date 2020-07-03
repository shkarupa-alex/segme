from tensorflow.keras import Sequential
from tensorflow.keras import initializers, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>DexiNed')
class UpConvBlock(layers.Layer):
    def __init__(self, up_scale, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.up_scale = up_scale
        self.output_filters = 16

    @shape_type_conversion
    def build(self, input_shape):
        total_up_scale = 2 ** self.up_scale
        rand_init = initializers.random_normal(stddev=0.01)
        trunc_init0 = initializers.TruncatedNormal()
        trunc_init1 = initializers.TruncatedNormal(stddev=0.1)

        self.features = Sequential()
        for i in range(self.up_scale):
            is_last = i == self.up_scale - 1
            out_features = 1 if is_last else self.output_filters
            kernel_init0 = trunc_init0 if is_last else rand_init
            kernel_init1 = trunc_init1 if is_last else rand_init

            self.features.add(layers.Conv2D(
                filters=out_features,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='relu',
                kernel_initializer=kernel_init0))
            self.features.add(layers.Conv2DTranspose(
                out_features,
                kernel_size=(total_up_scale, total_up_scale),
                strides=2,
                padding='same',
                kernel_initializer=kernel_init1))

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
