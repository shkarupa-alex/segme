import numpy as np
from keras.src import ops
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.metrics import reduction_metrics
from keras.src.saving import register_keras_serializable

from segme.ops import convert_image_dtype
from segme.ops import squared_difference


@register_keras_serializable(package="SegMe>Metric>Matting")
class Grad(reduction_metrics.Sum):
    def __init__(self, sigma=1.4, name="grad", dtype=None):
        """Creates a `GradientError` instance for matting task (by default
        downscales input by 255).

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.sigma = sigma

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = convert_image_dtype(y_true, self.dtype)
        y_pred = convert_image_dtype(y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)

        y_pred, y_true = squeeze_or_expand_to_same_rank(
            y_pred, y_true, sample_weight
        )
        if sample_weight is not None:
            y_pred_rank = len(y_pred.shape)
            sample_weight_rank = len(sample_weight.shape)

            if y_pred_rank == sample_weight_rank + 1:
                sample_weight = ops.expand_dims(sample_weight, axis=-1)

        values = gradient_error(y_true, y_pred, self.sigma, sample_weight)

        return super().update_state(values)

    def result(self):
        return super().result() * 0.001

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma})

        return config


def _togray(inputs):
    axis_hwc = list(range(1, inputs.shape.ndims))
    minval = ops.min(inputs, axis=axis_hwc, keepdims=True)
    maxval = ops.max(inputs, axis=axis_hwc, keepdims=True)

    return ops.divide_no_nan(inputs - minval, maxval - minval)


def _gauss(inputs, sigma):
    return np.exp(-(inputs**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _dgauss(inputs, sigma):
    return -inputs * _gauss(inputs, sigma) / (sigma**2)


def _gauss_filter(sigma, epsilon=1e-2):
    half = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))
    ).astype(np.int32)
    size = 2 * half + 1

    arange = np.arange(size)
    kernel0 = _gauss(arange[:, None] - half, sigma)
    kernel1 = _dgauss(arange[None, :] - half, sigma)

    kernel0 = kernel0 / np.sqrt(np.sum(kernel0**2))
    kernel1 = kernel1 / np.sqrt(np.sum(kernel1**2))

    return kernel0, kernel1, size


def _gauss_gradient(inputs, size, kernel_x, kernel_y):
    pad_size = max(size - 1, 0)
    pad_after = pad_size // 2
    pad_before = pad_size - pad_after

    # Replicate padding (matlab style)
    row_before = ops.repeat(inputs[:, :1, ...], pad_before, axis=1)
    row_afters = ops.repeat(inputs[:, -1:, ...], pad_before, axis=1)
    inputs = ops.concatenate([row_before, inputs, row_afters], axis=1)
    col_before = ops.repeat(inputs[:, :, :1, ...], pad_before, axis=2)
    col_afters = ops.repeat(inputs[:, :, -1:, ...], pad_before, axis=2)
    inputs = ops.concatenate([col_before, inputs, col_afters], axis=2)

    grad_x = ops.depthwise_conv(inputs, kernel_x[0], strides=1, padding="valid")
    grad_x = ops.depthwise_conv(grad_x, kernel_x[1], strides=1, padding="valid")

    grad_y = ops.depthwise_conv(inputs, kernel_y[0], strides=1, padding="valid")
    grad_y = ops.depthwise_conv(grad_y, kernel_y[1], strides=1, padding="valid")

    return grad_x, grad_y


def gradient_error(y_true, y_pred, sigma, sample_weight=None):
    channels = y_pred.shape[-1]
    if channels is None:
        raise ValueError(
            "Channel dimension of the predictions should be defined. "
            "Found `None`."
        )

    y_pred = _togray(y_pred)
    y_true = _togray(y_true)

    kernel0, kernel1, size = _gauss_filter(sigma)
    kernel0 = np.tile(kernel0[..., None, None], (1, 1, channels, 1))
    kernel1 = np.tile(kernel1[..., None, None], (1, 1, channels, 1))
    kernel0 = kernel0.astype(y_pred.dtype.as_numpy_dtype)
    kernel1 = kernel1.astype(y_pred.dtype.as_numpy_dtype)

    kernel_x = ops.cast(kernel0, y_pred.dtype), ops.cast(kernel1, y_pred.dtype)
    kernel_y = kernel0.transpose([1, 0, 2, 3]), kernel1.transpose([1, 0, 2, 3])
    kernel_y = ops.cast(kernel_y[0], y_pred.dtype), ops.cast(
        kernel_y[1], y_pred.dtype
    )

    y_pred_x, y_pred_y = _gauss_gradient(y_pred, size, kernel_x, kernel_y)
    y_true_x, y_true_y = _gauss_gradient(y_true, size, kernel_x, kernel_y)

    pred_amp = ops.sqrt(y_pred_x**2 + y_pred_y**2)
    true_amp = ops.sqrt(y_true_x**2 + y_true_y**2)

    result = squared_difference(pred_amp, true_amp)

    if sample_weight is not None:
        result *= sample_weight

    axis_hwc = list(range(1, result.shape.ndims))
    result = ops.sum(result, axis=axis_hwc)

    return result
