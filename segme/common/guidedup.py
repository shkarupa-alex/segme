import tensorflow as tf
from keras import layers
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.interrough import BilinearInterpolation
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Common')
class BoxFilter(layers.Layer):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.radius = radius

    def call(self, inputs, **kwargs):
        outputs = tf.cumsum(inputs, axis=1)

        left = outputs[:, self.radius:2 * self.radius + 1]
        middle = outputs[:, 2 * self.radius + 1:] - outputs[:, :-2 * self.radius - 1]
        right = outputs[:, -1:] - outputs[:, -2 * self.radius - 1:-self.radius - 1]
        outputs = tf.concat([left, middle, right], axis=1)

        outputs = tf.cumsum(outputs, axis=2)

        left = outputs[:, :, self.radius:2 * self.radius + 1]
        middle = outputs[:, :, 2 * self.radius + 1:] - outputs[:, :, :-2 * self.radius - 1]
        right = outputs[:, :, -1:] - outputs[:, :, -2 * self.radius - 1:-self.radius - 1]

        outputs = tf.concat([left, middle, right], axis=2)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'radius': self.radius})

        return config


@register_keras_serializable(package='SegMe>Common')
class GuidedFilter(layers.Layer):
    """ Proposed in: https://arxiv.org/abs/1803.05619 """

    def __init__(self, radius=4, filters=64, kernel_size=1, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, dtype='uint8'),  # image
            layers.InputSpec(ndim=4)]  # target

        self.radius = radius
        self.filters = filters
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    @shape_type_conversion
    def build(self, input_shape):
        channels = [shape[-1] for shape in input_shape]
        if None in channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, dtype='uint8', axes={-1: channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: channels[1]})
        ]

        self.intbysample = BilinearInterpolation(None)
        self.guide = Sequential([
            ConvNormAct(self.filters, self.kernel_size, name='guide_cna'),
            layers.Conv2D(channels[1], self.kernel_size, padding='same', name='guide_proj')
        ], name='guide')
        self.box = BoxFilter(self.radius, name='box')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        images, targets = inputs
        targets = self.intbysample([targets, images])

        normals = preprocess_input(tf.cast(images, self.compute_dtype), mode='tf')
        guides = self.guide(normals)

        size = tf.ones_like(targets, dtype=self.compute_dtype)
        size = self.box(size)

        guides_mean = self.box(guides) / size
        targets_mean = self.box(targets) / size

        covariance = self.box(guides * targets) / size - guides_mean * targets_mean
        variance = self.box(guides ** 2) / size - guides_mean ** 2

        scale = covariance / (variance + self.epsilon)
        bias = targets_mean - scale * guides_mean

        scale_mean = self.box(scale) / size
        bias_mean = self.box(bias) / size

        outputs = scale_mean * guides + bias_mean

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + input_shape[1][-1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'radius': self.radius,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'epsilon': self.epsilon
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class ConvGuidedFilter(layers.Layer):
    """ Proposed in: https://arxiv.org/abs/1803.05619 """

    def __init__(self, radius=1, filters=32, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, dtype='uint8'),  # image high
            layers.InputSpec(ndim=4)  # target low
        ]

        self.radius = radius
        self.filters = filters
        self.kernel_size = kernel_size

    @shape_type_conversion
    def build(self, input_shape):
        channels = [shape[-1] for shape in input_shape]
        if None in channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, dtype='uint8', axes={-1: channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: channels[1]})]

        self.interpolate = BilinearInterpolation(None)

        self.guide = Sequential([
            ConvNormAct(self.filters, self.kernel_size, name='guide_cna'),
            layers.Conv2D(channels[1], self.kernel_size, padding='same', name='guide_proj')
        ], name='guide')
        self.box = layers.DepthwiseConv2D(
            3, padding='same', dilation_rate=self.radius, use_bias=False, kernel_initializer='ones', name='box')
        self.conva = Sequential([
            ConvNormAct(self.filters, 1, name='conva_cna0'),
            ConvNormAct(self.filters, 1, name='conva_cna1'),
            layers.Conv2D(channels[1], 1, use_bias=False, name='conva_proj')
        ], name='conva')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        images_high, targets_low = inputs

        normals_high = preprocess_input(tf.cast(images_high, self.compute_dtype), mode='tf')
        normals_low = self.interpolate([normals_high, targets_low])

        guides_high = self.guide(normals_high)
        guides_low = self.guide(normals_low)

        size = tf.ones_like(targets_low, dtype=self.compute_dtype)
        size = self.box(size)

        guides_mean = self.box(guides_low) / size
        targets_mean = self.box(targets_low) / size

        covariance = self.box(guides_low * targets_low) / size - guides_mean * targets_mean
        variance = self.box(guides_low ** 2) / size - guides_mean ** 2

        scale = self.conva(tf.concat([covariance, variance], axis=-1))
        bias = targets_mean - scale * guides_mean

        scale_mean = self.interpolate([scale, images_high])
        bias_mean = self.interpolate([bias, images_high])

        outputs = scale_mean * guides_high + bias_mean

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + input_shape[1][-1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'radius': self.radius,
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })

        return config
