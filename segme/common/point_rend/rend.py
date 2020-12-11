import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion, smart_cond
from .head import PointHead
from .sample import classification_uncertainty, uncertain_points_with_randomness
from .sample import point_sample, uncertain_points_coords_on_grid
from ..head import HeadActivation
from ..resizebysample import resize_by_sample
from ..resizebyscale import resize_by_scale


@utils.register_keras_serializable(package='SegMe>PointRend')
class PointRend(layers.Layer):
    def __init__(self, classes, units, points, oversample, importance, fines, residual, align_corners, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # images
            layers.InputSpec(ndim=4, axes={-1: classes})  # coarse features
        ]
        self.input_spec += [layers.InputSpec(ndim=4) for _ in range(fines)]  # fine grained features

        if fines < 1:
            raise ValueError('At least one fine grained feature map required')

        self.classes = classes
        self.units = units
        self.points = normalize_tuple(points, 2, 'points')
        self.oversample = oversample
        self.importance = importance
        self.residual = residual
        self.fines = fines
        self.align_corners = align_corners

    @shape_type_conversion
    def build(self, input_shape):
        self.point_head = PointHead(classes=self.classes, units=self.units, fines=self.fines, residual=self.residual)
        self.head_act = HeadActivation(self.classes)

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()

        image_shape, coarse_shape = tf.shape(inputs[0]), tf.shape(inputs[1])
        assertions = [
            tf.debugging.assert_less_equal(
                coarse_shape[1] * 2, image_shape[1],
                message='Coarse feature maps height should be at least twice smaller then images one'),
            tf.debugging.assert_less_equal(
                coarse_shape[2] * 2, image_shape[2],
                message='Coarse feature maps width should be at least twice smaller then images one'),
        ]
        with tf.control_dependencies(assertions):
            predict_logits, point_logits, point_coords = smart_cond(
                training, lambda: self._train(inputs), lambda: self._eval(inputs))

            predict_logits = resize_by_sample([predict_logits, inputs[0]], align_corners=self.align_corners)
            predict_probs = self.head_act(predict_logits)

            return predict_probs, point_logits, point_coords

    def _train(self, inputs):
        input_images, coarse_features, *fine_features = inputs
        coarse_features = tf.cast(coarse_features, 'float32')

        point_coords = uncertain_points_with_randomness(
            coarse_features, points=self.points[0], oversample=self.oversample, importance=self.importance)
        point_coords = tf.stop_gradient(point_coords)

        coarse_points = point_sample([coarse_features, point_coords])
        fine_points = [point_sample([ff, point_coords]) for ff in fine_features]
        point_logits = self.point_head([coarse_points, *fine_points])

        return coarse_features, point_logits, point_coords

    def _eval(self, inputs):
        input_images, coarse_features, *fine_features = inputs

        image_shape, coarse_shape = tf.shape(input_images), tf.shape(coarse_features)
        subdivisions = tf.cast(tf.round(tf.maximum(
            image_shape[1] / coarse_shape[1], image_shape[2] / coarse_shape[2])), 'int32')

        _, predict_logits, point_logits, point_coords = tf.while_loop(
            lambda it, cf, pl, pc: tf.less(it, subdivisions),
            lambda it, cf, pl, pc: self._subdiv(it, cf, fine_features, pl, pc),
            [
                0,  # subdivision counter
                tf.cast(coarse_features, 'float32'),  # coarse features
                tf.zeros((coarse_shape[0], 0, self.classes), 'float32'),  # point logits
                tf.zeros((coarse_shape[0], 0, 2), 'float32')  # point coords
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None, None, self.classes]),
                tf.TensorShape([None, None, self.classes]),
                tf.TensorShape([None, None, 2])
            ])

        return predict_logits, point_logits, point_coords

    def _subdiv(self, iteration, coarse_features, fine_features, prev_logits, prev_coords):
        coarse_features = resize_by_scale(coarse_features, scale=2, align_corners=self.align_corners)

        point_indices, point_coords = uncertain_points_coords_on_grid(coarse_features, points=self.points[1])

        coarse_points = point_sample([coarse_features, point_coords])
        fine_points = [point_sample([ff, point_coords]) for ff in fine_features]
        point_logits = self.point_head([coarse_points, *fine_points])

        logits_shape = tf.shape(coarse_features)
        logits_batch, logits_height, logits_width = logits_shape[0], logits_shape[1], logits_shape[2]
        flat_logits = tf.reshape(coarse_features, [logits_batch, logits_height * logits_width, -1])

        indices_shape = tf.shape(point_indices)
        batch_size, point_size = indices_shape[0], indices_shape[1]
        batch_idx = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1))
        point_batches = tf.tile(batch_idx, (1, point_size, 1))
        update_indices = tf.concat([point_batches, tf.expand_dims(point_indices, axis=-1)], axis=-1)

        flat_logits = tf.tensor_scatter_nd_update(flat_logits, update_indices, point_logits)
        coarse_features = tf.reshape(flat_logits, [logits_batch, logits_height, logits_width, self.classes])

        point_logits = layers.concatenate([prev_logits, point_logits], axis=1, dtype=prev_logits.dtype)
        point_coords = layers.concatenate([prev_coords, point_coords], axis=1, dtype=prev_logits.dtype)

        return iteration + 1, coarse_features, point_logits, point_coords

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (input_shape[0][:-1] + (self.classes,)), \
               (input_shape[0][0], None, self.classes), \
               (input_shape[0][0], None, 2)

    def compute_output_signature(self, input_signature):
        expected_dtypes = ['float32'] * 3
        output_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype=dt, shape=ds.shape) for dt, ds in zip(expected_dtypes, output_signature)]

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'units': self.units,
            'points': self.points,
            'oversample': self.oversample,
            'importance': self.importance,
            'fines': self.fines,
            'residual': self.residual,
            'align_corners': self.align_corners
        })

        return config
