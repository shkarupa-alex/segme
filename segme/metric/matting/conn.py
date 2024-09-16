import numpy as np
from keras.src import ops
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.metrics import reduction_metrics
from keras.src.saving import register_keras_serializable
from tfmiss.image import connected_components

from segme.ops import convert_image_dtype


@register_keras_serializable(package="SegMe>Metric>Matting")
class Conn(reduction_metrics.Sum):
    def __init__(self, step=0.1, name="conn", dtype=None):
        """Creates a `ConnectivityError` instance for matting task (by default
        downscales input by 255).

        Args:
            step: (Optional) float percents for threshold step estimating
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.step = step

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

        values = connectivity_error(y_true, y_pred, self.step, sample_weight)

        return super().update_state(values)

    def result(self):
        return super().result() * 0.001

    def get_config(self):
        config = super().get_config()
        config.update({"step": self.step})

        return config


def connectivity_error(y_true, y_pred, step, sample_weight=None):
    thresh_steps = list(np.arange(step, step + 1.0, step))

    true_shape = ops.shape(y_true)
    batch, height, width, channels = true_shape
    minmax_len = height * width * channels + 2

    thresh_map = []
    for threshold in thresh_steps:
        combined_input = (y_true >= threshold) & (y_pred >= threshold)
        component_labels = connected_components(combined_input, normalize=False)

        squezed_labels = ops.transpose(component_labels, [0, 3, 1, 2])
        squezed_labels = ops.reshape(squezed_labels, [batch * channels, -1])

        component_sizes = ops.bincount(squezed_labels, minlength=minmax_len)
        component_sizes = ops.reshape(component_sizes, [batch, channels, -1])
        component_sizes *= ops.concatenate(
            [
                ops.zeros([batch, channels, 1], dtype=component_sizes.dtype),
                ops.ones_like(component_sizes)[..., 1:],
            ],
            axis=-1,
        )

        component_max = ops.argmax(component_sizes, axis=-1)[..., None, None]
        component_max = ops.transpose(component_max, [0, 2, 3, 1])
        component_max = ops.cast(component_max, component_labels.dtype)

        component_back = component_labels != component_max
        thresh_map.append(component_back)

    thresh_map.append(ops.ones_like(y_true, dtype="bool"))

    thresh_map = ops.stack(thresh_map, axis=-1)
    thresh_map = ops.reshape(thresh_map, [batch, -1, len(thresh_steps) + 1])
    thresh_map = ops.cast(ops.argmax(thresh_map, axis=-1), y_true.dtype) * step
    thresh_map = ops.reshape(thresh_map, true_shape)

    pred_d = y_pred - thresh_map
    true_d = y_true - thresh_map
    pred_phi = 1.0 - pred_d * ops.cast(pred_d >= 0.15, y_pred.dtype)
    true_phi = 1.0 - true_d * ops.cast(true_d >= 0.15, y_true.dtype)

    result = ops.abs(pred_phi - true_phi)

    if sample_weight is not None:
        result *= sample_weight

    axis_hwc = list(range(1, result.shape.ndims))
    result = ops.sum(result, axis=axis_hwc)

    return result
