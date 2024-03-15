import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.adppool import AdaptiveAveragePooling
from segme.common.convnormact import ConvNormAct
from segme.common.sequence import Sequence
from segme.common.resize import BilinearInterpolation


@register_keras_serializable(package='SegMe>Common')
class PyramidPooling(layers.Layer):
    def __init__(self, filters, sizes=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.sizes = sizes

    @shape_type_conversion
    def build(self, input_shape):
        self.stages = [Sequence([
            AdaptiveAveragePooling(size, name='pool'),
            ConvNormAct(self.filters, 1, name='cna')
        ], name=f'stage_{size}') for size in self.sizes]
        self.interpolate = BilinearInterpolation(None)
        self.bottleneck = ConvNormAct(self.filters, 3, name='bottleneck')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [stage(inputs) for stage in self.stages]
        outputs = [self.interpolate([output, inputs]) for output in outputs]
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
            'sizes': self.sizes
        })

        return config
