import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class ReceptiveField(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.branch0 = ConvNormAct(self.filters, 1)
        self.branch1 = Sequential([
            ConvNormAct(self.filters, 1),
            ConvNormAct(self.filters, (1, 3)),
            ConvNormAct(self.filters, (3, 1)),
            ConvNormAct(self.filters, 3, dilation_rate=3)
        ])
        self.branch2 = Sequential([
            ConvNormAct(self.filters, 1),
            ConvNormAct(self.filters, (1, 5)),
            ConvNormAct(self.filters, (5, 1)),
            ConvNormAct(self.filters, 3, dilation_rate=5)
        ])
        self.branch3 = Sequential([
            ConvNormAct(self.filters, 1),
            ConvNormAct(self.filters, (1, 7)),
            ConvNormAct(self.filters, (7, 1)),
            ConvNormAct(self.filters, 3, dilation_rate=7)
        ])
        self.conv = ConvNormAct(self.filters, 3)
        self.proj = ConvNormAct(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.concat([
            self.branch0(inputs), self.branch1(inputs), self.branch2(inputs), self.branch3(inputs)], axis=-1)
        outputs = self.conv(outputs)
        outputs += self.proj(inputs)
        outputs = tf.nn.relu(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
