import tensorflow as tf
from keras import Sequential, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu, SameConv, resize_by_sample


@register_keras_serializable(package='SegMe>CascadePSP')
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
            ConvNormRelu(self.filters, 3),
            SameConv(self.filters, 3),
        ])
        self.conv2 = Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            ConvNormRelu(self.filters, 3),
            SameConv(self.filters, 3),
        ])
        self.shortcut = SameConv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        high, low = inputs

        high = resize_by_sample([high, low])
        outputs = self.conv1(tf.concat([high, low], axis=-1))
        short = self.shortcut(high)
        outputs = outputs + short
        delta = self.conv2(outputs)
        outputs = outputs + delta

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
