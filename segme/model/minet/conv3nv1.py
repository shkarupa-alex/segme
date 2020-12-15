from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import resize_by_sample


@utils.register_keras_serializable(package='SegMe>MINet')
class Conv3nV1(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels_h = input_shape[0][-1]
        self.channels_m = input_shape[1][-1]
        self.channels_l = input_shape[2][-1]
        if self.channels_h is None or self.channels_m is None or self.channels_l is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels_h}),
            layers.InputSpec(ndim=4, axes={-1: self.channels_m}),
            layers.InputSpec(ndim=4, axes={-1: self.channels_l})
        ]

        min_channels = min(self.channels_h, self.channels_m, self.channels_l)

        self.relu = layers.ReLU()
        self.pool = layers.AveragePooling2D(2, strides=2)

        # stage 0
        self.cbr_hh0 = Sequential([
            layers.Conv2D(min_channels, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.cbr_mm0 = Sequential([
            layers.Conv2D(min_channels, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.cbr_ll0 = Sequential([
            layers.Conv2D(min_channels, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # stage 1
        self.conv_hh1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_hm1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_mh1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_mm1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_ml1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_lm1 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_ll1 = layers.Conv2D(min_channels, 3, padding='same')
        self.bn_h1 = layers.BatchNormalization()
        self.bn_m1 = layers.BatchNormalization()
        self.bn_l1 = layers.BatchNormalization()

        # stage 2
        self.conv_hm2 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_lm2 = layers.Conv2D(min_channels, 3, padding='same')
        self.conv_mm2 = layers.Conv2D(min_channels, 3, padding='same')
        self.bn_m2 = layers.BatchNormalization()

        # stage 3
        self.conv_mm3 = layers.Conv2D(self.filters, 3, padding='same')
        self.bn_m3 = layers.BatchNormalization()

        self.identity = layers.Conv2D(self.filters, 1, padding='same')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_h, inputs_m, inputs_l = inputs

        # stage 0
        h = self.cbr_hh0(inputs_h)
        m = self.cbr_mm0(inputs_m)
        l = self.cbr_ll0(inputs_l)

        # stage 1
        h2h = self.conv_hh1(h)
        m2h = self.conv_mh1(resize_by_sample([m, h2h], method='nearest', align_corners=False))

        h2m = self.conv_hm1(self.pool(h))
        m2m = self.conv_mm1(m)
        l2m = self.conv_lm1(resize_by_sample([l, m2m], method='nearest', align_corners=False))

        m2l = self.conv_ml1(self.pool(m))
        l2l = self.conv_ll1(l)

        h = self.relu(self.bn_h1(layers.add([h2h, m2h])))
        m = self.relu(self.bn_m1(layers.add([
            resize_by_sample([h2m, m2m], method='nearest', align_corners=False), m2m, l2m])))
        l = self.relu(self.bn_l1(layers.add([
            resize_by_sample([m2l, l2l], method='nearest', align_corners=False), l2l])))

        # stage 2
        h2m = self.conv_hm2(self.pool(h))
        m2m = self.conv_mm2(m)
        l2m = self.conv_lm2(resize_by_sample([l, m2m], method='nearest', align_corners=False))
        m = self.relu(self.bn_m2(layers.add([
            resize_by_sample([h2m, m2m], method='nearest', align_corners=False), m2m, l2m])))

        # stage 3
        out = layers.add([self.bn_m3(self.conv_mm3(m)), self.identity(inputs_m)])

        return self.relu(out)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
