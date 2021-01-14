import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>PointRend')
class ClassificationUncertainty(layers.Layer):
    def __init__(self, from_logits=True, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=2)
        self.from_logits = from_logits

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: self.channels})

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.from_logits:
            activation = tf.nn.softmax if self.channels > 1 else tf.nn.sigmoid
            inputs = activation(inputs)

        if 1 == self.channels:
            inputs = layers.concatenate([1. - inputs, inputs])

        scores, _ = tf.math.top_k(inputs, k=2)
        uncertainty = scores[..., 1] - scores[..., 0]

        return uncertainty

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype=input_signature.dtype, shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({'from_logits': self.from_logits})

        return config


def classification_uncertainty(inputs, **kwargs):
    return ClassificationUncertainty(**kwargs)(inputs)


@utils.register_keras_serializable(package='SegMe>PointRend')
class PointSample(layers.Layer):
    def __init__(self, align_corners, mode='bilinear', **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),
            layers.InputSpec(ndim=3, axes={-1: 2})
        ]

        if mode not in {'bilinear', 'nearest'}:
            raise ValueError('Wrong interpolation mode. Only "bilinear" and "nearest" supported')
        self.align_corners = align_corners
        self.mode = mode

    def call(self, inputs, **kwargs):
        features, grid = inputs

        assertions = [
            tf.debugging.assert_greater_equal(
                tf.reduce_min(grid), tf.cast(0.0, grid.dtype), message='Grid values should be in range [0; 1]'),
            tf.debugging.assert_less_equal(
                tf.reduce_max(grid), tf.cast(1.0, grid.dtype), message='Grid values should be in range [0; 1]')
        ]
        with tf.control_dependencies(assertions):
            features_shape, grid_shape = tf.shape(features), tf.shape(grid)
            safe_features = tf.pad(features, [[0, 0], [1, 1], [1, 1], [0, 0]])
            safe_features = tf.cast(safe_features, grid.dtype)
            grid = tf.reverse(grid, axis=[-1])
            size = tf.cast(features_shape[1:3], grid.dtype)

            if self.align_corners:
                grid = grid * (size - 1)
            else:
                grid = (2.0 * grid * size - 1) / 2

            batch_size, point_size = grid_shape[0], grid_shape[1]
            batch_idx = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1))
            coord_batches = tf.tile(batch_idx, (1, point_size, 1))
            coord_bounds = features_shape[1:3] + 1

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
            if not (features_dtype.is_floating or features_dtype.is_complex):
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
            'align_corners': self.align_corners,
            'mode': self.mode
        })

        return config


def point_sample(inputs, **kwargs):
    return PointSample(**kwargs)(inputs)


@utils.register_keras_serializable(package='SegMe>PointRend')
class UncertainPointsWithRandomness(layers.Layer):
    def __init__(self, points, align_corners, oversample, importance, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if not 0. <= points <= 1.:
            raise ValueError('Parameter "points" should be in range [0; 1]')
        if oversample < 1.:
            raise ValueError('Parameter "oversample" should be greater or equal 1')
        if not 0. <= importance <= 1.:
            raise ValueError('Parameter "importance" should be in range [0; 1]')

        self.points = float(points)
        self.align_corners = align_corners
        self.oversample = float(oversample)
        self.importance = float(importance)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        total_points = tf.cast(input_shape[1] * input_shape[2], self.compute_dtype) * self.points

        sampled_size = tf.cast(total_points * self.oversample, 'int32')
        point_coords = tf.random.uniform((batch_size, sampled_size, 2), dtype=self.compute_dtype)
        point_logits = point_sample([inputs, point_coords], align_corners=self.align_corners)

        # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
        # Calculating uncertainties of the coarse predictions first and sampling them for points leads
        # to incorrect results.
        # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
        # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
        # However, if we calculate uncertainties for the coarse predictions first,
        # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.

        point_uncerts = classification_uncertainty(point_logits)
        uncertain_size = tf.cast(total_points * self.importance, 'int32')
        random_size = tf.maximum(0, tf.cast(total_points, 'int32') - uncertain_size)

        _, top_indices = tf.math.top_k(point_uncerts, k=uncertain_size)
        top_points = tf.gather(point_coords, top_indices, batch_dims=1)

        point_coords = layers.concatenate([
            top_points,
            tf.random.uniform((batch_size, random_size, 2), dtype=self.compute_dtype)
        ], axis=1, dtype=self.compute_dtype)

        return point_coords

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        total_points = None
        if input_shape[1] is not None and input_shape[2] is not None:
            total_points = int(input_shape[1] * input_shape[2] * self.points)

        return input_shape[:1] + (total_points, 2)

    def get_config(self):
        config = super().get_config()
        config.update({
            'points': self.points,
            'align_corners': self.align_corners,
            'oversample': self.oversample,
            'importance': self.importance
        })

        return config


def uncertain_points_with_randomness(inputs, **kwargs):
    return UncertainPointsWithRandomness(**kwargs)(inputs)


@utils.register_keras_serializable(package='SegMe>PointRend')
class UncertainPointsCoordsOnGrid(layers.Layer):
    def __init__(self, points, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if not 0. <= points <= 1.:
            raise ValueError('Parameter "points" should be in range [0; 1]')

        self.points = float(points)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        input_height = tf.cast(input_shape[1], self.compute_dtype)
        input_width = tf.cast(input_shape[2], self.compute_dtype)

        total_points = tf.cast(input_height * input_width * self.points, 'int32')
        uncert_map = classification_uncertainty(inputs)
        flat_inputs = tf.reshape(uncert_map, [batch_size, -1])
        _, top_indices = tf.math.top_k(flat_inputs, k=total_points)

        exp_indices = tf.expand_dims(tf.cast(top_indices, self.compute_dtype), axis=-1)
        point_coords = tf.concat([
            0.5 / input_width + (exp_indices % input_width) / input_width,
            0.5 / input_height + (exp_indices // input_width) / input_height
        ], axis=-1)

        return top_indices, point_coords

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        total_points = None
        if input_shape[1] is not None and input_shape[2] is not None:
            total_points = int(input_shape[1] * input_shape[2] * self.points)

        return input_shape[:1] + (total_points,), input_shape[:1] + (total_points, 2)

    def compute_output_signature(self, input_signature):
        expected_dtypes = ['int32', input_signature.dtype]
        output_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype=dt, shape=ds.shape) for dt, ds in zip(expected_dtypes, output_signature)]

    def get_config(self):
        config = super().get_config()
        config.update({'points': self.points})

        return config


def uncertain_points_coords_on_grid(inputs, **kwargs):
    return UncertainPointsCoordsOnGrid(**kwargs)(inputs)
