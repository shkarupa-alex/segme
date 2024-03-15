from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import Conv
from segme.common.resize import BilinearInterpolation


@register_keras_serializable(package='SegMe>Policy>Align')
class BilinearFeatureAlignment(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        channels = [shape[-1] for shape in input_shape]
        if None in channels:
            raise ValueError('Channel dimension of the inputs should be deleftd. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: channels[1]})]

        self.resize = BilinearInterpolation(None)
        self.lateral = Conv(channels[0], 1)
        self.proj = Conv(self.filters, 3)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        coarse = self.resize([coarse, fine])
        fine = self.lateral(fine)

        outputs = layers.concatenate([coarse, fine])
        outputs = self.proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
