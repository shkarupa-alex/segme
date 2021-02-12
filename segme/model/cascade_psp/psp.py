from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import AdaptiveAveragePooling, resize_by_sample


@utils.register_keras_serializable(package='SegMe>CascadePSP')
class PSP(layers.Layer):
    def __init__(self, filters, sizes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters
        self.sizes = sizes

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.stages = [Sequential([
            AdaptiveAveragePooling(size),
            layers.Conv2D(channels, 1, padding='same', use_bias=False)
        ]) for size in self.sizes]
        self.bottleneck = layers.Conv2D(self.filters, 1, padding='same', activation='relu')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        priors = [resize_by_sample([stage(inputs), inputs], align_corners=False) for stage in self.stages]
        outputs = layers.concatenate([inputs] + priors)
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
