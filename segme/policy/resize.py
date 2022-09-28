import itertools
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.align.impf import SpatialEncoding
from segme.common.convnormact import ConvNormAct
from segme.common.impfunc import make_coords, query_features
from segme.common.interrough import NearestInterpolation, BilinearInterpolation
from segme.common.sequent import Sequential
from segme.policy.registry import LayerRegistry

RESIZERS = LayerRegistry()
RESIZERS.register('inter_linear')(BilinearInterpolation)


@RESIZERS.register('inter_liif')
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


@RESIZERS.register('inter_griif')
@register_keras_serializable(package='SegMe>Policy>Resize')
class GrIIFInterpolation(NearestInterpolation):
    def __init__(self, scale=None, multi_scale=False, learn_positions=True, symmetric_pad=True, **kwargs):
        super().__init__(scale=scale, **kwargs)

        self.multi_scale = multi_scale
        self.learn_positions = learn_positions
        self.symmetric_pad = symmetric_pad

    @shape_type_conversion
    def build(self, input_shape):
        if 1. == self.scale:
            return super().build(input_shape)

        channels = input_shape[-1] if self.scale is not None else input_shape[0][-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.imnet = Sequential([ConvNormAct(channels, 1), ConvNormAct(channels, 1)])
        self.relnet = layers.Dense(9, activation='softmax')

        if self.learn_positions:
            self.posnet = SpatialEncoding()

        super().build(input_shape)

    def resize(self, inputs, size):
        if self.imnet is None:
            return inputs

        batch_size, features_height, features_width, _ = tf.unstack(tf.shape(inputs))
        targets_height, targets_width = tf.unstack(size)
        target_volume = batch_size * targets_height * targets_width
        features_height_ = tf.cast(features_height, 'float32')
        features_width_ = tf.cast(features_width, 'float32')
        targets_height_ = tf.cast(targets_height, 'float32')
        targets_width_ = tf.cast(targets_width, 'float32')

        height_scale = features_height_ / targets_height_
        width_scale = features_width_ / targets_width_

        features_hrange = tf.range(features_height_, dtype='float32')
        features_wrange = tf.range(features_width_, dtype='float32')
        features_hwmesh = tf.meshgrid(features_hrange, features_wrange, indexing='ij')
        features_coords = tf.stack(features_hwmesh, axis=-1)
        features_coords = tf.image.resize(features_coords, [targets_height, targets_width], method='nearest')

        targets_hrange = tf.range(targets_height_, dtype='float32')
        targets_hrange = (targets_hrange + 0.5) * height_scale - 0.5
        targets_wrange = tf.range(targets_width_, dtype='float32')
        targets_wrange = (targets_wrange + 0.5) * width_scale - 0.5
        targets_hwmesh = tf.meshgrid(targets_hrange, targets_wrange, indexing='ij')
        targets_coords = tf.stack(targets_hwmesh, axis=-1)

        relative_coords = (targets_coords - features_coords)
        relative_coords = tf.tile(relative_coords[None, :, :, None], [batch_size, 1, 1, 9, 1])
        relative_coords += tf.cast(list(itertools.product([-1., 0, 1.], [-1., 0, 1.])), 'float32')
        relative_coords = tf.reshape(relative_coords, [target_volume, 3, 3, 2])
        relative_coords = tf.stop_gradient(relative_coords)

        pad_mode = 'SYMMETRIC' if self.symmetric_pad else 'CONSTANT'
        feature_patches = tf.pad(inputs, [(0, 0), (1, 1), (1, 1), (0, 0)], mode=pad_mode)
        feature_patches = tf.image.extract_patches(
            feature_patches, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        feature_patches = tf.image.resize(feature_patches, (targets_height, targets_width), method='nearest')
        feature_patches = tf.reshape(feature_patches, [target_volume, 3, 3, inputs.shape[-1]])

        if self.learn_positions:
            relative_positions = self.posnet(relative_coords)
        else:
            relative_positions = tf.cast(relative_coords, self.compute_dtype)

        full_features = [feature_patches, relative_positions]
        if self.multi_scale:
            full_features.append(tf.fill([target_volume, 3, 3, 1], tf.cast(height_scale, self.compute_dtype)))
            full_features.append(tf.fill([target_volume, 3, 3, 1], tf.cast(width_scale, self.compute_dtype)))
        full_features = tf.concat(full_features, axis=-1)

        output_positions = tf.reshape(relative_positions, [target_volume, 9 * relative_positions.shape[-1]])
        output_positions = self.relnet(output_positions)
        output_positions = tf.reshape(output_positions, [target_volume, 3, 3, 1])

        output_features = self.imnet(full_features)
        output_features *= output_positions
        output_features = tf.reduce_sum(output_features, axis=[1, 2])
        output_features = tf.reshape(output_features, [batch_size, targets_height, targets_width, inputs.shape[-1]])

        return output_features

    def get_config(self):
        config = super().get_config()
        config.update({
            'multi_scale': self.multi_scale,
            'learn_positions': self.learn_positions,
            'symmetric_pad': self.symmetric_pad
        })

        return config


@RESIZERS.register('inter_lgriif')
@register_keras_serializable(package='SegMe>Policy>Resize')
class LGrIIFInterpolation(NearestInterpolation):
    def __init__(self, scale=None, multi_scale=False, learn_positions=True, symmetric_pad=True, **kwargs):
        super().__init__(scale=scale, **kwargs)

        self.multi_scale = multi_scale
        self.learn_positions = learn_positions
        self.symmetric_pad = symmetric_pad

    @shape_type_conversion
    def build(self, input_shape):
        if 1. == self.scale:
            return super().build(input_shape)

        channels = input_shape[-1] if self.scale is not None else input_shape[0][-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.imnet = Sequential([ConvNormAct(channels, 3, padding='valid'), ConvNormAct(channels, 1)])

        if self.learn_positions:
            self.posnet = SpatialEncoding()

        super().build(input_shape)

    def resize(self, inputs, size):
        if self.imnet is None:
            return inputs

        batch_size, features_height, features_width, _ = tf.unstack(tf.shape(inputs))
        targets_height, targets_width = tf.unstack(size)
        target_volume = batch_size * targets_height * targets_width
        features_height_ = tf.cast(features_height, 'float32')
        features_width_ = tf.cast(features_width, 'float32')
        targets_height_ = tf.cast(targets_height, 'float32')
        targets_width_ = tf.cast(targets_width, 'float32')

        height_scale = features_height_ / targets_height_
        width_scale = features_width_ / targets_width_

        features_hrange = tf.range(features_height_, dtype='float32')
        features_wrange = tf.range(features_width_, dtype='float32')
        features_hwmesh = tf.meshgrid(features_hrange, features_wrange, indexing='ij')
        features_coords = tf.stack(features_hwmesh, axis=-1)
        features_coords = tf.image.resize(features_coords, [targets_height, targets_width], method='nearest')

        targets_hrange = tf.range(targets_height_, dtype='float32')
        targets_hrange = (targets_hrange + 0.5) * height_scale - 0.5
        targets_wrange = tf.range(targets_width_, dtype='float32')
        targets_wrange = (targets_wrange + 0.5) * width_scale - 0.5
        targets_hwmesh = tf.meshgrid(targets_hrange, targets_wrange, indexing='ij')
        targets_coords = tf.stack(targets_hwmesh, axis=-1)

        relative_coords = (targets_coords - features_coords)
        relative_coords = tf.tile(relative_coords[None, :, :, None], [batch_size, 1, 1, 9, 1])
        relative_coords += tf.cast(list(itertools.product([-1., 0, 1.], [-1., 0, 1.])), 'float32')
        relative_coords = tf.reshape(relative_coords, [target_volume, 3, 3, 2])
        relative_coords = tf.stop_gradient(relative_coords)

        pad_mode = 'SYMMETRIC' if self.symmetric_pad else 'CONSTANT'
        feature_patches = tf.pad(inputs, [(0, 0), (1, 1), (1, 1), (0, 0)], mode=pad_mode)
        feature_patches = tf.image.extract_patches(
            feature_patches, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        feature_patches = tf.image.resize(feature_patches, (targets_height, targets_width), method='nearest')
        feature_patches = tf.reshape(feature_patches, [target_volume, 3, 3, inputs.shape[-1]])

        if self.learn_positions:
            relative_positions = self.posnet(relative_coords)
        else:
            relative_positions = tf.cast(relative_coords, self.compute_dtype)

        full_features = [feature_patches, relative_positions]
        if self.multi_scale:
            full_features.append(tf.fill([target_volume, 3, 3, 1], tf.cast(height_scale, self.compute_dtype)))
            full_features.append(tf.fill([target_volume, 3, 3, 1], tf.cast(width_scale, self.compute_dtype)))
        full_features = tf.concat(full_features, axis=-1)

        output_features = self.imnet(full_features)
        output_features = tf.reshape(output_features, [batch_size, targets_height, targets_width, inputs.shape[-1]])

        return output_features

    def get_config(self):
        config = super().get_config()
        config.update({
            'multi_scale': self.multi_scale,
            'learn_positions': self.learn_positions,
            'symmetric_pad': self.symmetric_pad
        })

        return config
