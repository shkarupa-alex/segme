from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import resize_by_sample


@utils.register_keras_serializable(package='SegMe>CascadePSP')
class Upsample(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.conv1 = Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.filters, 3, padding='same'),
        ])
        self.conv2 = Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.filters, 3, padding='same'),
        ])
        self.shortcut = layers.Conv2D(self.filters, 1, padding='same')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        high, low = inputs

        high = resize_by_sample([high, low], align_corners=False)
        outputs = self.conv1(layers.concatenate([high, low]))
        short = self.shortcut(high)
        outputs = layers.add([outputs, short])
        delta = self.conv2(outputs)
        outputs = layers.add([outputs, delta])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
