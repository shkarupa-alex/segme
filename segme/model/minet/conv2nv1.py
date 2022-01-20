import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu, resize_by_sample


@register_keras_serializable(package='SegMe>MINet')
class Conv2nV1(layers.Layer):
    def __init__(self, filters, main, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]

        if main not in {0, 1}:
            raise ValueError('Parameter "main" should equals 0 or 1')
        self.filters = filters
        self.main = main

    @shape_type_conversion
    def build(self, input_shape):
        self.channels_h = input_shape[0][-1]
        self.channels_l = input_shape[1][-1]
        if self.channels_h is None or self.channels_l is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels_h}),
            layers.InputSpec(ndim=4, axes={-1: self.channels_l})
        ]

        min_channels = min(self.channels_h, self.channels_l)

        self.relu = layers.ReLU()
        self.pool = layers.AveragePooling2D(2, strides=2, padding='same')

        # stage 0
        self.cbr_hh0 = ConvNormRelu(min_channels, 3, padding='same')
        self.cbr_ll0 = ConvNormRelu(min_channels, 3, padding='same')

        # stage 1
        self.conv_hh1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_hl1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_lh1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_ll1 = layers.Conv2D(min_channels, 3, padding='same')
        self.bn_l1 = layers.BatchNormalization()
        self.bn_h1 = layers.BatchNormalization()

        if self.main == 0:
            # stage 2
            self.conv_hh2 = layers.Conv2D(min_channels, 3, padding='same')
            self.conv_lh2 = layers.Conv2D(min_channels, 3, padding='same')
            self.bn_h2 = layers.BatchNormalization()

            # stage 3
            self.conv_hh3 = layers.Conv2D(self.filters, 3, padding='same')
            self.bn_h3 = layers.BatchNormalization()

            self.identity = layers.Conv2D(self.filters, 1, padding='same')

        elif self.main == 1:
            # stage 2
            self.conv_hl2 = layers.Conv2D(min_channels, 3, padding='same')
            self.conv_ll2 = layers.Conv2D(min_channels, 3, padding='same')
            self.bn_l2 = layers.BatchNormalization()

            # stage 3
            self.conv_ll3 = layers.Conv2D(self.filters, 3, padding='same')
            self.bn_l3 = layers.BatchNormalization()

            self.identity = layers.Conv2D(self.filters, 1, padding='same')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_h, inputs_l = inputs

        # stage 0
        h = self.cbr_hh0(inputs_h)
        l = self.cbr_ll0(inputs_l)

        # stage 1
        h2h = self.conv_hh1(h)
        h2l = self.conv_hl1(self.pool(h))
        l2l = self.conv_ll1(l)
        l2h = self.conv_lh1(resize_by_sample([l, h2h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        h = self.relu(self.bn_h1(layers.add([h2h, l2h])))
        l = self.relu(self.bn_l1(layers.add([l2l, h2l])))

        if self.main == 0:
            # stage 2
            h2h = self.conv_hh2(h)
            l2h = self.conv_lh2(resize_by_sample([l, h2h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            h_fuse = self.relu(self.bn_h2(layers.add([h2h, l2h])))

            # stage 3
            out = layers.add([self.bn_h3(self.conv_hh3(h_fuse)), self.identity(inputs_h)])
        else:  # self.main == 1
            # stage 2
            h2l = self.conv_hl2(self.pool(h))
            l2l = self.conv_ll2(l)
            l_fuse = self.relu(self.bn_l2(layers.add([h2l, l2l])))

            # stage 3
            out = layers.add([self.bn_l3(self.conv_ll3(l_fuse)), self.identity(inputs_l)])

        return self.relu(out)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[self.main][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'main': self.main
        })

        return config
