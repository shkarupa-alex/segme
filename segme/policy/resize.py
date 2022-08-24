import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.align.impf import SpatialEncoding
from segme.common.convnormact import ConvNormAct
from segme.common.impfunc import make_coords, query_features
from segme.common.internear import NearestInterpolation
from segme.common.sequent import Sequential
from segme.registry import LayerRegistry

RESIZERS = LayerRegistry()
RESIZERS.register('inter_linearcomp')({
    'class_name': 'SegMe>Policy>Resize>BilinearInterpolation', 'config': {'compat': True}})
RESIZERS.register('inter_liif9')({
    'class_name': 'SegMe>Policy>Resize>LIIFInterpolation', 'config': {'feat_unfold': 3, 'local_ensemble': False}})


@RESIZERS.register('inter_linear')
@register_keras_serializable(package='SegMe>Policy>Resize')
class BilinearInterpolation(NearestInterpolation):
    def __init__(self, scale=None, compat=False, **kwargs):
        super().__init__(scale=scale, **kwargs)

        self.compat = compat

    def resize(self, inputs, size):
        if self.compat:
            outputs = tf.compat.v1.image.resize(
                inputs, size, method=tf.compat.v1.image.ResizeMethod.BILINEAR, align_corners=True)
        else:
            outputs = tf.image.resize(inputs, size, method=tf.image.ResizeMethod.BILINEAR)

        if inputs.dtype != outputs.dtype:
            outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'compat': self.compat})

        return config


@RESIZERS.register('inter_liif4')
@register_keras_serializable(package='SegMe>Policy>Resize')
class LIIFInterpolation(NearestInterpolation):
    def __init__(self, scale=None, feat_unfold=False, local_ensemble=True, learn_positions=True, symmetric_pad=True,
                 **kwargs):
        super().__init__(scale=scale, **kwargs)

        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.learn_positions = learn_positions
        self.symmetric_pad = symmetric_pad

    @shape_type_conversion
    def build(self, input_shape):
        self.imnet, self.posnet = None, None
        if 1. != self.scale:
            channels = input_shape[-1] if self.scale is not None else input_shape[0][-1]
            if channels is None:
                raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

            self.imnet = Sequential([ConvNormAct(channels, 1), ConvNormAct(channels, 1)])
            if self.learn_positions:
                self.posnet = SpatialEncoding()

        super().build(input_shape)

    def resize(self, inputs, size):
        coords = make_coords([tf.shape(inputs)[0], *tf.unstack(size)])
        outputs = query_features(
            inputs, coords, self.imnet, posnet=self.posnet, cells=None, feat_unfold=self.feat_unfold,
            local_ensemble=self.local_ensemble, symmetric_pad=self.symmetric_pad)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'feat_unfold': self.feat_unfold,
            'local_ensemble': self.local_ensemble,
            'learn_positions': self.learn_positions,
            'symmetric_pad': self.symmetric_pad
        })

        return config
