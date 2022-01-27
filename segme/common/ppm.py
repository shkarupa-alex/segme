import tensorflow as tf
from keras import activations, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .adppool import AdaptiveAveragePooling
from .convnormrelu import ConvNormRelu
from .resizebysample import resize_by_sample


@register_keras_serializable(package='SegMe')
class PyramidPooling(layers.Layer):
    def __init__(self, filters, sizes, activation='relu', standardized=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.sizes = sizes
        self.activation = activations.get(activation)
        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.stages = [models.Sequential([
            AdaptiveAveragePooling(size),
            ConvNormRelu(self.filters, 1, activation=self.activation, standardized=self.standardized)])
            for size in self.sizes]
        self.bottleneck = ConvNormRelu(self.filters, 3, activation=self.activation, standardized=self.standardized)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [stage(inputs) for stage in self.stages]
        outputs = [resize_by_sample([o, inputs]) for o in outputs]
        outputs = tf.concat([inputs] + outputs, axis=-1)
        outputs = self.bottleneck(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'sizes': self.sizes,
            'activation': activations.serialize(self.activation),
            'standardized': self.standardized
        })

        return config
