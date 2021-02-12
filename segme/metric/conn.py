from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import SumOverBatchSize, metrics_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow_addons.image import connected_components


@tf.keras.utils.register_keras_serializable(package='SegMe')
class Conn(SumOverBatchSize):
    def __init__(self, divider=255., step=0.1, name='conn', dtype=None):
        """Creates a `ConnectivityError` instance for matting task (by default downscales input by 255).

        Args:
            divider: A float value for input scaling.
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.divider = divider
        self.step = step

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype) / self.divider
        y_pred = tf.cast(y_pred, self._dtype) / self.divider
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)

        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)
        y_pred, y_true, sample_weight = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)

        values = connectivity_error(y_true, y_pred, self.step, sample_weight)

        return super().update_state(values)

    def get_config(self):
        config = super().get_config()
        config.update({
            'divider': self.divider,
            'step': self.step
        })

        return config


def connectivity_error(y_true, y_pred, step, sample_weight=None):
    thresh_steps = list(np.arange(step, step + 1., step))

    true_shape = tf.shape(y_true)
    batch_size = true_shape[0]
    squezed_shape = true_shape[:3]
    pred_squezed = tf.reshape(y_pred, squezed_shape)
    true_squezed = tf.reshape(y_true, squezed_shape)

    thresh_map = []
    for threshold in thresh_steps:
        combined_input = (true_squezed >= threshold) & (pred_squezed >= threshold)
        component_labels = connected_components(combined_input)

        squezed_labels = tf.reshape(component_labels, [batch_size, -1])
        squezed_weights = tf.cast(squezed_labels != 0, y_true.dtype)

        component_sizes = tf.math.bincount(squezed_labels, axis=-1, weights=squezed_weights)
        component_max = tf.argmax(component_sizes, axis=-1, output_type='int32')[:, None, None]

        component_back = component_labels != component_max
        thresh_map.append(component_back[..., None])
    thresh_map.append(tf.ones_like(y_true, dtype='bool'))

    thresh_map = tf.concat(thresh_map, axis=-1)
    thresh_map = tf.reshape(thresh_map, [batch_size, -1, len(thresh_steps) + 1])
    thresh_map = tf.cast(tf.argmax(thresh_map, axis=-1), y_true.dtype) * step
    thresh_map = tf.reshape(thresh_map, true_shape)

    pred_d = y_pred - thresh_map
    true_d = y_true - thresh_map
    pred_phi = 1. - pred_d * tf.cast(pred_d >= 0.15, y_pred.dtype)
    true_phi = 1. - true_d * tf.cast(true_d >= 0.15, y_true.dtype)

    result = tf.abs(pred_phi - true_phi)

    if sample_weight is not None:
        result *= sample_weight

    axes = list(range(1, result.shape.ndims))
    result = tf.reduce_sum(result, axis=axes)

    return result
