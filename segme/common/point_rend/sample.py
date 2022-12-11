import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.head import ClassificationActivation
from segme.common.impfunc import grid_sample


@register_keras_serializable(package='SegMe>Common>PointRend')
class ClassificationUncertainty(layers.Layer):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=2)
        self.from_logits = from_logits

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: self.channels})

        if self.from_logits:
            self.class_act = ClassificationActivation(dtype=self.compute_dtype)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.from_logits:
            inputs = self.class_act(inputs)

        if 1 == self.channels:
            inputs = layers.concatenate([1. - inputs, inputs])

        scores, _ = tf.math.top_k(inputs, k=2)
        uncertainty = scores[..., 1] - scores[..., 0]

        return uncertainty

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super().get_config()
        config.update({'from_logits': self.from_logits})

        return config


@register_keras_serializable(package='SegMe>Common>PointRend')
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

        outputs = grid_sample(
            features, grid[:, None] * 2. - 1., mode=self.mode, align_corners=self.align_corners)[:, 0]

        return outputs

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


@register_keras_serializable(package='SegMe>Common>PointRend')
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

    @shape_type_conversion
    def build(self, input_shape):
        self.class_uncert = ClassificationUncertainty()
        self.point_sample = PointSample(align_corners=self.align_corners)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        total_points = tf.cast(input_shape[1] * input_shape[2], 'float32') * self.points

        sampled_size = tf.cast(total_points * self.oversample, 'int32')
        point_coords = tf.random.uniform((batch_size, sampled_size, 2), dtype=self.compute_dtype)
        point_logits = self.point_sample([inputs, point_coords])

        # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
        # Calculating uncertainties of the coarse predictions first and sampling them for points leads
        # to incorrect results.
        # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
        # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
        # However, if we calculate uncertainties for the coarse predictions first,
        # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.

        point_uncerts = self.class_uncert(point_logits)
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


@register_keras_serializable(package='SegMe>Common>PointRend')
class UncertainPointsCoordsOnGrid(layers.Layer):
    def __init__(self, points, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if not 0. <= points <= 1.:
            raise ValueError('Parameter "points" should be in range [0; 1]')

        self.points = float(points)

    @shape_type_conversion
    def build(self, input_shape):
        self.class_uncert = ClassificationUncertainty()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, height, width, _ = tf.unstack(tf.shape(inputs))
        height = tf.cast(height, 'float32')
        width = tf.cast(width, 'float32')

        total_points = height * width * self.points
        sampled_size = tf.cast(total_points, 'int32')

        uncert_map = self.class_uncert(inputs)
        flat_inputs = tf.reshape(uncert_map, [batch, -1])
        _, top_indices = tf.math.top_k(flat_inputs, k=sampled_size)

        exp_indices = tf.cast(top_indices, 'float32')[..., None]
        point_coords = tf.concat([
            0.5 / width + (exp_indices % width) / width,
            0.5 / height + (exp_indices // width) / height
        ], axis=-1)
        point_coords = tf.cast(point_coords, self.compute_dtype)

        return top_indices, point_coords

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        total_points = None
        if input_shape[1] is not None and input_shape[2] is not None:
            total_points = int(input_shape[1] * input_shape[2] * self.points)

        return input_shape[:1] + (total_points,), input_shape[:1] + (total_points, 2)

    def compute_output_signature(self, input_signature):
        expected_dtypes = ['int32', self.compute_dtype]
        output_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype=dt, shape=ds.shape) for dt, ds in zip(expected_dtypes, output_signature)]

    def get_config(self):
        config = super().get_config()
        config.update({'points': self.points})

        return config
