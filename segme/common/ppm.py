import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.adppool import AdaptiveAveragePooling
from segme.common.convnormact import ConvNormAct
from segme.common.sequent import Sequential
from segme.common.interrough import NearestInterpolation
from segme.common.intersmooth import SmoothInterpolation


@register_keras_serializable(package='SegMe>Common')
class PyramidPooling(layers.Layer):
    def __init__(self, filters, sizes=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.sizes = sizes

    @shape_type_conversion
    def build(self, input_shape):
        self.stages = [
            Sequential([AdaptiveAveragePooling(size), ConvNormAct(self.filters, 1)]) for size in self.sizes]
        self.interpolations = [
            NearestInterpolation(None) if 1 == size else SmoothInterpolation(None) for size in self.sizes]
        self.bottleneck = ConvNormAct(self.filters, 3)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [stage(inputs) for stage in self.stages]
        outputs = [interpolate([output, inputs]) for interpolate, output in zip(self.interpolations, outputs)]
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
