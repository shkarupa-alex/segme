import math
import tensorflow as tf
from keras import initializers, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.adppool import AdaptiveAveragePooling
from segme.common.align.impf import SpatialEncoding
from segme.common.align.fade import CarafeConvolution
from segme.common.intersmooth import SmoothInterpolation
from segme.common.convnormact import ConvNormAct, ConvAct, Conv, Act
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Common>Align>CAFF')
class CaffFeatureAlignment(layers.Layer):
    def __init__(self, filters, pool_size, kernel_size=5, reduce_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.filters = filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.reduce_ratio = reduce_ratio

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be deleftd. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: self.channels[1]})]

        coarse_filters = math.ceil(self.channels[1] * self.reduce_ratio / 8) * 8
        self.coarse_select = SeFeatureSelection(coarse_filters)

        fine_filters = math.ceil(self.channels[0] * self.reduce_ratio / 8) * 8
        self.fine_select = GuidedFeatureSelection(fine_filters, self.pool_size)

        pred_kernel = max(3, self.kernel_size - 2)
        self.up_kernel = ImplicitKernelPrediction(self.kernel_size ** 2, pred_kernel)
        self.carafe_up = CarafeConvolution(self.kernel_size)

        self.fuse_proj = ConvNormAct(self.filters, 3)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        coarse_selected = self.coarse_select(coarse)
        fine_selected = self.fine_select([fine, coarse_selected])

        upsample_kernel = self.up_kernel([fine_selected, coarse_selected])
        coarse_upsampled = self.carafe_up([coarse_selected, upsample_kernel])

        outputs = tf.concat([fine_selected, coarse_upsampled], axis=-1)
        outputs = self.fuse_proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'pool_size': self.pool_size,
            'kernel_size': self.kernel_size,
            'reduce_ratio': self.reduce_ratio
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class SeFeatureSelection(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        self.se = Sequential([
            layers.GlobalAvgPool2D(keepdims=True),
            ConvAct(self.filters, 1, kernel_initializer='variance_scaling'),
            layers.Conv2D(self.channels, 1, activation='sigmoid', kernel_initializer='variance_scaling')])

        self.proj = Conv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        attention = self.se(inputs)
        outputs = inputs * (attention + 1.)  # same as inputs * attention + inputs = SE + skip connection
        outputs = self.proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config


@register_keras_serializable(package='SegMe>Common')
class GuidedFeatureSelection(layers.Layer):
    def __init__(self, filters, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]

        self.filters = filters
        self.pool_size = pool_size

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be deleftd. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: self.channels[1]})]

        self.sample_proj = Conv(None, 3)
        self.guide_proj = Conv(None, 3)
        self.avg_pool = AdaptiveAveragePooling(self.pool_size)

        self.att_bias = self.add_weight(
            'bias',
            shape=[1, self.channels[0], self.channels[1]],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)

        self.out_proj = Conv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        sample, guide = inputs

        sample_small = self.sample_proj(sample)
        sample_small = self.avg_pool(sample_small)
        sample_small = tf.reshape(sample_small, [-1, self.pool_size ** 2, self.channels[0]])

        guide_small = self.guide_proj(guide)
        guide_small = self.avg_pool(guide_small)
        guide_small = tf.reshape(guide_small, [-1, self.pool_size ** 2, self.channels[1]])

        scale = 1. / self.pool_size
        attention = tf.matmul(sample_small * scale, guide_small, transpose_a=True)
        attention += self.att_bias
        attention = tf.nn.sigmoid(attention)
        attention = attention[:, None, None]
        attention = tf.reduce_mean(attention, axis=-1)

        outputs = sample * (attention + 1.)  # same as inputs * attention + inputs = attention + skip connection
        outputs = self.out_proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'pool_size': self.pool_size
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class ImplicitKernelPrediction(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # fine
            layers.InputSpec(ndim=4)]  # coarse

        self.filters = filters
        self.kernel_size = kernel_size

    @shape_type_conversion
    def build(self, input_shape):
        channels = [shape[-1] for shape in input_shape]
        if None in channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: channels[0]}), layers.InputSpec(ndim=4, axes={-1: channels[1]})]

        self.intbysample = SmoothInterpolation(None)
        self.content = Conv(self.filters, self.kernel_size)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        coarse = self.intbysample([coarse, fine])

        outputs = tf.concat([fine, coarse], axis=-1)
        outputs = self.content(outputs)
        outputs = tf.nn.softmax(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })

        return config
