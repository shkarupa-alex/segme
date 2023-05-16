from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.carafe import CarafeConvolution
from segme.common.resize import NearestInterpolation


@register_keras_serializable(package='SegMe>Policy>Align>FADE')
class FadeFeatureAlignment(layers.Layer):
    """
    Proposed in "FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling"
    https://arxiv.org/pdf/2207.10392.pdf
    """

    def __init__(self, filters, kernel_size=5, embedding_size=64, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.filters = filters
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size

    @shape_type_conversion
    def build(self, input_shape):
        self.intnear = NearestInterpolation(None)

        self.gate = layers.Conv2D(1, 1, activation='sigmoid')
        self.kernel = SemiShift(self.kernel_size ** 2, max(3, self.kernel_size - 2), self.embedding_size)
        self.carafe = CarafeConvolution(self.kernel_size)

        # in original version coarse and fine features should have same channel size
        # here we project them to the same channel size before merging
        self.fine = layers.Conv2D(self.filters, 1)
        self.coarse = layers.Conv2D(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        gate = self.gate(coarse)
        gate = self.intnear([gate, fine])

        kernel = self.kernel([fine, coarse])
        coarse = self.carafe([coarse, kernel])
        coarse = self.coarse(coarse)

        fine = self.fine(coarse)

        outputs = gate * fine + (1. - gate) * coarse

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'embedding_size': self.embedding_size
        })

        return config


@register_keras_serializable(package='SegMe>Policy>Align>FADE')
class SemiShift(layers.Layer):
    def __init__(self, filters, kernel_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.filters = filters
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size

    @shape_type_conversion
    def build(self, input_shape):
        self.fine = layers.Conv2D(self.embedding_size, 1)
        self.coarse = layers.Conv2D(self.embedding_size, 1, use_bias=False)
        self.content = layers.Conv2D(self.filters, self.kernel_size, padding='same')

        self.internear = NearestInterpolation()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        fine = self.fine(fine)
        fine = self.content(fine)

        coarse = self.coarse(coarse)
        coarse = self.content(coarse)
        coarse = self.internear([coarse, fine])

        outputs = fine + coarse

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'embedding_size': self.embedding_size
        })

        return config
