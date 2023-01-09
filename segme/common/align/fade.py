import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.interrough import NearestInterpolation
from segme.common.patchxla import extract_patches_xla


@register_keras_serializable(package='SegMe>Common>Align>FADE')
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


@register_keras_serializable(package='SegMe>Common>Align>FADE')
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
        self.coarse = layers.Conv2D(self.embedding_size, 1)
        self.fine = layers.Conv2D(self.embedding_size, 1, use_bias=False)
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
        outputs = tf.nn.softmax(outputs, axis=-1)

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


@register_keras_serializable(package='SegMe>Common>Align>FADE')
class CarafeConvolution(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features
            layers.InputSpec(ndim=4)]  # mask

        self.kernel_size = kernel_size

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: self.channels[1]})]

        self.internear = NearestInterpolation()

        self.group_size = self.channels[1] // (self.kernel_size ** 2)
        if self.group_size < 1 or self.channels[1] != self.group_size * self.kernel_size ** 2:
            raise ValueError('Wrong mask channel dimension.')

        if self.channels[0] % self.group_size:
            raise ValueError('Unable to split features into groups.')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        features, masks = inputs

        batch, height, width, _ = tf.unstack(tf.shape(masks))
        output_shape = self.compute_output_shape([features.shape, masks.shape])

        features = extract_patches_xla(
            features, [1, self.kernel_size, self.kernel_size, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
        features = self.internear([features, masks])

        features = tf.reshape(
            features,
            (batch, height, width, self.group_size, self.channels[0] // self.group_size, self.kernel_size ** 2))
        masks = tf.reshape(masks, (batch, height, width, self.group_size, 1, self.kernel_size ** 2))

        outputs = tf.matmul(features, masks, transpose_b=True)

        outputs = tf.reshape(outputs, (batch, height, width, self.channels[0]))
        outputs.set_shape(output_shape)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.channels[0],)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})

        return config
