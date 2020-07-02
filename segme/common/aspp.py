import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .atsepconv import AtrousSepConv2D
from .upbysample import up_by_sample_2d


@utils.register_keras_serializable(package='SegMe')
class ASPPPool2D(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.pool = Sequential([
            layers.GlobalAveragePooling2D(),
            # (batch, channels) -> (batch, 1, 1, channels)
            layers.Lambda(lambda pooled: tf.expand_dims(tf.expand_dims(pooled, 1), 1)),
            layers.Conv2D(self.filters, 1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.pool(inputs)
        outputs = up_by_sample_2d([outputs, inputs])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config


@utils.register_keras_serializable(package='SegMe')
class ASPP2D(layers.Layer):
    _stride_rates = {
        8: [12, 24, 36],
        16: [6, 12, 18],
        32: [3, 6, 9]
    }

    def __init__(self, filters, stride, **kwargs):
        # TODO
        # When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
        # When using 'xception_65' or 'resnet_v1' model variants, we set
        # atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters
        self.stride = stride
        if stride not in self._stride_rates:
            raise NotImplementedError('Unsupported output stride')

    @shape_type_conversion
    def build(self, input_shape):
        rate0, rate1, rate2 = self._stride_rates[self.stride]
        self.conv3r0 = AtrousSepConv2D(self.filters, rate0, name='aspp1')
        self.conv3r1 = AtrousSepConv2D(self.filters, rate1, name='aspp2')
        self.conv3r2 = AtrousSepConv2D(self.filters, rate2, name='aspp3')

        self.conv1 = Sequential([
            layers.Conv2D(self.filters, 1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='aspp0')
        self.pool = ASPPPool2D(self.filters, name='aspp4')
        self.proj = Sequential([
            layers.Conv2D(self.filters, 1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.1)  # 0.5 in some implementations
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = layers.concatenate([
            self.conv1(inputs),
            self.conv3r0(inputs),
            self.conv3r1(inputs),
            self.conv3r2(inputs),
            self.pool(inputs)
        ])
        outputs = self.proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'stride': self.stride
        })

        return config
