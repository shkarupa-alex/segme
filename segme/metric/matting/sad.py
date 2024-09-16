from keras.src import ops
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.metrics import reduction_metrics
from keras.src.saving import register_keras_serializable

from segme.ops import convert_image_dtype


@register_keras_serializable(package="SegMe>Metric>Matting")
class SAD(reduction_metrics.Sum):
    def __init__(self, name="sad", dtype=None):
        """Creates a `SumAbsoluteDifference` instance for matting task (by
        default downscales input by 255).

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = convert_image_dtype(y_true, self.dtype)
        y_pred = convert_image_dtype(y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)

        y_pred, y_true = squeeze_or_expand_to_same_rank(
            y_pred, y_true, sample_weight
        )
        if sample_weight is not None:
            y_pred_rank = ops.ndim(y_pred)
            sample_weight_rank = ops.ndim(sample_weight)

            if y_pred_rank == sample_weight_rank + 1:
                sample_weight = ops.expand_dims(sample_weight, axis=-1)

        values = sum_absolute_difference(y_true, y_pred, sample_weight)

        return super().update_state(values)

    def result(self):
        return super().result() * 0.001


def sum_absolute_difference(y_true, y_pred, sample_weight=None):
    result = ops.abs(y_pred - y_true)

    if sample_weight is not None:
        result *= sample_weight

    axis_hwc = list(range(1, result.shape.ndims))
    result = ops.sum(result, axis=axis_hwc)

    return result
