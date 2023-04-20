import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.src.metrics import SumOverBatchSize
from keras.src.utils import losses_utils, metrics_utils


@register_keras_serializable(package='SegMe>Metric>Matting')
class SAD(SumOverBatchSize):
    def __init__(self, name='sad', dtype=None):
        """Creates a `SumAbsoluteDifference` instance for matting task (by default downscales input by 255).

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        dtype_true = tf.dtypes.as_dtype(y_true.dtype)
        scale_true = dtype_true.max if dtype_true.is_integer else 1.
        y_true = tf.cast(y_true, self._dtype) / scale_true

        dtype_pred = tf.dtypes.as_dtype(y_pred.dtype)
        scale_pred = dtype_pred.max if dtype_pred.is_integer else 1.
        y_pred = tf.cast(y_pred, self._dtype) / scale_pred

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)

        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)

        if sample_weight is None:
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)
        else:
            y_pred, y_true, sample_weight = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)

        values = sum_absolute_difference(y_true, y_pred, sample_weight)

        return super().update_state(values)

    def result(self):
        return super().result() / 1000.


def sum_absolute_difference(y_true, y_pred, sample_weight=None):
    result = tf.abs(y_pred - y_true)

    if sample_weight is not None:
        result *= sample_weight

    axis_hwc = list(range(1, result.shape.ndims))
    result = tf.reduce_sum(result, axis=axis_hwc)

    return result
