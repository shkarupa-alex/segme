from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.carafe import CarafeConvolution
from segme.common.resize import NearestInterpolation


@register_keras_serializable(package="SegMe>Policy>Align>FADE")
class FadeFeatureAlignment(layers.Layer):
    """
    Proposed in "FADE: Fusing the Assets of Decoder and Encoder for
    Task-Agnostic Upsampling"
    https://arxiv.org/pdf/2207.10392.pdf
    """

    def __init__(self, filters, kernel_size=5, embedding_size=64, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # fine
            InputSpec(ndim=4),
        ]  # coarse

        self.filters = filters
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.intnear = NearestInterpolation(None, dtype=self.dtype_policy)

        self.gate = layers.Conv2D(
            1, 1, activation="sigmoid", dtype=self.dtype_policy
        )
        self.gate.build(input_shape[1])

        self.kernel = SemiShift(
            self.kernel_size**2,
            max(3, self.kernel_size - 2),
            self.embedding_size,
            dtype=self.dtype_policy,
        )
        self.kernel.build(input_shape)

        self.carafe = CarafeConvolution(
            self.kernel_size, dtype=self.dtype_policy
        )
        self.carafe.build(
            [input_shape[1], input_shape[0][:-1] + (self.kernel_size**2,)]
        )

        self.coarse = layers.Conv2D(self.filters, 1, dtype=self.dtype_policy)
        self.coarse.build(input_shape[0][:-1] + (input_shape[1][-1],))

        # in original version coarse and fine features should have same channel
        # size, here we project them to the same channel size before merging
        self.fine = layers.Conv2D(self.filters, 1, dtype=self.dtype_policy)
        self.fine.build(input_shape[0][:-1] + (self.filters,))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        gate = self.gate(coarse)
        gate = self.intnear([gate, fine])

        kernel = self.kernel([fine, coarse])
        coarse = self.carafe([coarse, kernel])
        coarse = self.coarse(coarse)

        fine = self.fine(coarse)  # TODO

        outputs = gate * fine + (1.0 - gate) * coarse

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "embedding_size": self.embedding_size,
            }
        )

        return config


@register_keras_serializable(package="SegMe>Policy>Align>FADE")
class SemiShift(layers.Layer):  # https://github.com/poppinace/fade/issues/2
    def __init__(self, filters, kernel_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # fine
            InputSpec(ndim=4),
        ]  # coarse

        self.filters = filters
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.fine = layers.Conv2D(
            self.embedding_size, 1, dtype=self.dtype_policy
        )
        self.fine.build(input_shape[0])

        self.coarse = layers.Conv2D(
            self.embedding_size, 1, use_bias=False, dtype=self.dtype_policy
        )
        self.coarse.build(input_shape[1])

        self.content = layers.Conv2D(
            self.filters,
            self.kernel_size,
            padding="same",
            dtype=self.dtype_policy,
        )
        self.content.build(input_shape[0][:-1] + (self.embedding_size,))

        self.internear = NearestInterpolation(dtype=self.dtype_policy)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        fine = self.fine(fine)
        fine = self.content(fine)  # TODO: 2 conv without act?

        coarse = self.coarse(coarse)
        coarse = self.content(coarse)
        coarse = self.internear([coarse, fine])

        outputs = fine + coarse

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "embedding_size": self.embedding_size,
            }
        )

        return config
