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
            activation = 'softmax' if self.channels > 1 else 'sigmoid'
            inputs = layers.Activation(activation, dtype=inputs.dtype)(inputs)

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
    def __init__(self, mode='bilinear', **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),
            layers.InputSpec(ndim=3, axes={-1: 2})
        ]

        if mode not in {'bilinear', 'nearest'}:
            raise ValueError('Wrong interpolation mode. Only "bilinear" and "nearest" supported')
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
            grid = (2.0 * grid * tf.cast(features_shape[1:3], grid.dtype) - 1) / 2

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
        config.update({'mode': self.mode})

        return config


def point_sample(inputs, **kwargs):
    return PointSample(**kwargs)(inputs)


@utils.register_keras_serializable(package='SegMe>PointRend')
class UncertainPointsWithRandomness(layers.Layer):
    def __init__(self, points, oversample=3, importance=0.75, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if oversample < 1:
            raise ValueError('Parameter "oversample" should be greater or equal 1')
        if importance < 0 or importance > 1:
            raise ValueError('Parameter "importance" should be in range [0; 1]')

        self.points = points
        self.oversample = oversample
        self.importance = importance

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        sampled_size = int(self.points * self.oversample)

        point_coords = tf.random.uniform((batch_size, sampled_size, 2), dtype=inputs.dtype)
        point_logits = point_sample([inputs, point_coords])

        # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
        # Calculating uncertainties of the coarse predictions first and sampling them for points leads
        # to incorrect results.
        # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
        # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
        # However, if we calculate uncertainties for the coarse predictions first,
        # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.

        point_uncerts = classification_uncertainty(point_logits)
        uncertain_size = int(self.importance * self.points)
        random_size = self.points - uncertain_size

        _, top_indices = tf.math.top_k(point_uncerts, k=uncertain_size)
        top_points = tf.gather(point_coords, top_indices, batch_dims=1)

        if random_size > 0:
            point_coords = layers.concatenate([
                top_points,
                tf.random.uniform((batch_size, random_size, 2), dtype=inputs.dtype)
            ], axis=1, dtype=inputs.dtype)

        return point_coords

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.points, 2)

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype=input_signature.dtype, shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'points': self.points,
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
        self.points = points

    def call(self, inputs, **kwargs):
        uncertainty = classification_uncertainty(inputs)

        shape = tf.shape(uncertainty)
        batch, height, width = shape[0], shape[1], shape[2]

        points_size = tf.math.minimum(height * width, self.points)
        flat_inputs = tf.reshape(uncertainty, [batch, height * width])
        _, top_indices = tf.math.top_k(flat_inputs, k=points_size)

        float_width = tf.cast(width, inputs.dtype)
        float_height = tf.cast(height, inputs.dtype)

        exp_indices = tf.expand_dims(tf.cast(top_indices, inputs.dtype), axis=-1)
        point_coords = tf.concat([
            0.5 / float_width + (exp_indices % float_width) / float_width,
            0.5 / float_height + (exp_indices // float_width) / float_height
        ], axis=-1)

        return top_indices, point_coords

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        points_size = None
        if input_shape[1] is not None and input_shape[2] is not None:
            points_size = min(input_shape[1] * input_shape[2], self.points)

        return input_shape[:1] + (points_size,), input_shape[:1] + (points_size, 2)

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
