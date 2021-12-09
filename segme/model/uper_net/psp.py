from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvBnRelu, resize_by_sample, AdaptiveAveragePooling


@register_keras_serializable(package='SegMe>UPerNet')
class PSP(layers.Layer):
    def __init__(self, filters, sizes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters
        self.sizes = sizes

    @shape_type_conversion
    def build(self, input_shape):
        self.stages = [models.Sequential([
            AdaptiveAveragePooling(size),
            ConvBnRelu(self.filters, 1, use_bias=False)
        ]) for size in self.sizes]
        self.bottleneck = ConvBnRelu(self.filters, 3, use_bias=False)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [stage(inputs) for stage in self.stages]
        outputs = [resize_by_sample([out, inputs]) for out in outputs]
        outputs = layers.concatenate([inputs] + outputs)
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
