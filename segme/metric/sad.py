import tensorflow as tf
from keras.metrics import SumOverBatchSize, metrics_utils
from keras.utils import losses_utils
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe')
class SAD(SumOverBatchSize):
    def __init__(self, divider=255., name='sad', dtype=None):
        """Creates a `SumAbsoluteDifference` instance for matting task (by default downscales input by 255).

        Args:
            divider: A float value for input scaling.
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.divider = divider

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)

        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)

        if sample_weight is None:
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)
        else:
            y_pred, y_true, sample_weight = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)

        values = sum_absolute_difference(y_true, y_pred, sample_weight)

        return super().update_state(values / self.divider)

    def result(self):
        return super().result() / 1000.

    def get_config(self):
        config = super().get_config()
        config.update({'divider': self.divider})

        return config


def sum_absolute_difference(y_true, y_pred, sample_weight=None):
    result = tf.abs(y_pred - y_true)

    if sample_weight is not None:
        result *= sample_weight

    axis_hwc = list(range(1, result.shape.ndims))
    result = tf.reduce_sum(result, axis=axis_hwc)

    return result
