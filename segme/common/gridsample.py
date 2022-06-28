import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe')
class GridSample(layers.Layer):
    def __init__(self, mode='bilinear', align_corners=False, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4, axes={-1: 2})]

        if mode not in {'bilinear', 'nearest'}:
            raise ValueError('Wrong interpolation mode. Only "bilinear" and "nearest" supported')

        self.align_corners = align_corners
        self.mode = mode

    def call(self, inputs, **kwargs):
        features, grid = inputs

        features_shape = tf.shape(features)
        features_size = features_shape[1:3]
        batch_size, point_height, point_width, _ = tf.unstack(tf.shape(grid))

        assertions = [
            tf.debugging.assert_equal(
                features_shape[0], batch_size, message='Batch size should be the same for features and grid'),
            tf.debugging.assert_greater_equal(
                tf.reduce_min(grid), tf.cast(-1.0, grid.dtype), message='Grid values should be in range [-1; 1]'),
            tf.debugging.assert_less_equal(
                tf.reduce_max(grid), tf.cast(1.0, grid.dtype), message='Grid values should be in range [-1; 1]')
        ]
        with tf.control_dependencies(assertions):
            safe_features = tf.pad(features, [[0, 0], [1, 1], [1, 1], [0, 0]])
            safe_features = tf.cast(safe_features, grid.dtype)
            grid = tf.reverse(grid, axis=[-1])
            size = tf.cast(features_size, grid.dtype)

            if self.align_corners:
                grid = (grid + 1) * (size - 1) * 0.5
            else:
                grid = (grid + 1) * size * 0.5 - 0.5

            batch_idx = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
            coord_batches = tf.tile(batch_idx, (1, point_height, point_width, 1))
            coord_bounds = features_size + 1

            def _lookup(coords):
                coords = tf.clip_by_value(tf.cast(coords, 'int32') + 1, 0, coord_bounds)
                indices = tf.concat([coord_batches, coords], axis=-1)
                return tf.gather_nd(safe_features, indices)

            if 'bilinear' == self.mode:
                grid_nw = tf.math.floor(grid)
                grid_ne = grid_nw + [1, 0]
                grid_sw = grid_nw + [0, 1]
                grid_se = grid_nw + [1, 1]

                nw = tf.math.reduce_prod(grid_se - grid, axis=-1, keepdims=True)
                ne = tf.math.reduce_prod((grid_sw - grid) * [1, -1], axis=-1, keepdims=True)
                sw = tf.math.reduce_prod((grid_ne - grid) * [-1, 1], axis=-1, keepdims=True)
                se = tf.math.reduce_prod(grid - grid_nw, axis=-1, keepdims=True)

                result = tf.add_n([
                    _lookup(grid_nw) * nw,
                    _lookup(grid_ne) * ne,
                    _lookup(grid_sw) * sw,
                    _lookup(grid_se) * se])

            else:  # 'nearest' == self.mode
                result = _lookup(tf.math.round(grid))

            features_dtype = tf.dtypes.as_dtype(features.dtype)
            if features_dtype.is_integer:
                result = tf.round(result)

            return tf.cast(result, features.dtype)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        features_shape, grid_shape = input_shape

        return grid_shape[:-1] + features_shape[-1:]

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype=input_signature[0].dtype, shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'align_corners': self.align_corners
        })

        return config


def grid_sample(inputs, **kwargs):
    return GridSample(**kwargs)(inputs)
