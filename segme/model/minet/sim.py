from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import ResizeBySample


@utils.register_keras_serializable(package='SegMe>MINet')
class SIM(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        self.relu = layers.ReLU()
        self.pool = layers.AveragePooling2D(2, strides=2, padding='same')
        self.resize = ResizeBySample(align_corners=False)

        self.cbr_hh0 = Sequential([
            layers.Conv2D(self.channels, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.cbr_hl0 = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.conv_hh1 = layers.Conv2D(self.channels, 3, padding='same')
        self.conv_hl1 = layers.Conv2D(self.filters, 3, padding='same')
        self.conv_lh1 = layers.Conv2D(self.channels, 3, padding='same')
        self.conv_ll1 = layers.Conv2D(self.filters, 3, padding='same')
        self.bn_l1 = layers.BatchNormalization()
        self.bn_h1 = layers.BatchNormalization()

        self.conv_hh2 = layers.Conv2D(self.channels, 3, padding='same')
        self.conv_lh2 = layers.Conv2D(self.channels, 3, padding='same')
        self.bn_h2 = layers.BatchNormalization()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # first conv
        h = self.cbr_hh0(inputs)
        l = self.cbr_hl0(self.pool(inputs))

        # mid conv
        h2h = self.conv_hh1(h)
        h2l = self.conv_hl1(self.pool(h))
        l2l = self.conv_ll1(l)
        l2h = self.conv_lh1(self.resize([l, inputs]))
        h = self.relu(self.bn_h1(layers.add([h2h, l2h])))
        l = self.relu(self.bn_l1(layers.add([l2l, h2l])))

        # last conv
        h2h = self.conv_hh2(h)
        l2h = self.conv_lh2(self.resize([l, inputs]))
        h = self.relu(self.bn_h2(layers.add([h2h, l2h])))

        return h

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
