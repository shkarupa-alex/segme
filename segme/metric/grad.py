import numpy as np
import tensorflow as tf
from keras.metrics import SumOverBatchSize, metrics_utils
from keras.utils import losses_utils
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe')
class Grad(SumOverBatchSize):
    def __init__(self, sigma=1.4, name='grad', dtype=None):
        """Creates a `GradientError` instance for matting task (by default downscales input by 255).

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.sigma = sigma

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

        values = gradient_error(y_true, y_pred, self.sigma, sample_weight)

        return super().update_state(values)

    def result(self):
        return super().result() / 1000.

    def get_config(self):
        config = super().get_config()
        config.update({'sigma': self.sigma})

        return config


def _togray(inputs):
    axis_hwc = list(range(1, inputs.shape.ndims))
    minval = tf.reduce_min(inputs, axis=axis_hwc, keepdims=True)
    maxval = tf.reduce_max(inputs, axis=axis_hwc, keepdims=True)

    return tf.math.divide_no_nan(inputs - minval, maxval - minval)


def _gauss(inputs, sigma):
    return np.exp(-inputs ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def _dgauss(inputs, sigma):
    return -inputs * _gauss(inputs, sigma) / (sigma ** 2)


def _gauss_filter(sigma, epsilon=1e-2):
    half = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * half + 1

    kernel = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            kernel[i, j] = _gauss(i - half, sigma) * _dgauss(j - half, sigma)
    kernel = kernel / np.sqrt(np.sum(np.abs(kernel) * np.abs(kernel)))

    return kernel, size


def _gauss_gradient(inputs, size, kernel_x, kernel_y):
    pad_size = max(size - 1, 0)
    pad_after = pad_size // 2
    pad_before = pad_size - pad_after

    # Replicate padding (matlab style)
    row_before = tf.repeat(inputs[:, :1, ...], pad_before, axis=1)
    row_afters = tf.repeat(inputs[:, -1:, ...], pad_before, axis=1)
    inputs = tf.concat([row_before, inputs, row_afters], axis=1)
    col_before = tf.repeat(inputs[:, :, :1, ...], pad_before, axis=2)
    col_afters = tf.repeat(inputs[:, :, -1:, ...], pad_before, axis=2)
    inputs = tf.concat([col_before, inputs, col_afters], axis=2)

    grad_x = tf.nn.conv2d(inputs, kernel_x, strides=[1] * 4, padding='VALID')
    grad_y = tf.nn.conv2d(inputs, kernel_y, strides=[1] * 4, padding='VALID')

    return grad_x, grad_y


def gradient_error(y_true, y_pred, sigma, sample_weight=None):
    y_pred = _togray(y_pred)
    y_true = _togray(y_true)

    channels = y_pred.shape[-1]
    if channels is None:
        raise ValueError('Channel dimension of the predictions should be defined. Found `None`.')

    kernel, size = _gauss_filter(sigma)
    kernel = kernel.astype(y_pred.dtype.as_numpy_dtype)

    kernel_x = np.tile(kernel[..., None, None], (1, 1, channels, 1))
    kernel_x = tf.constant(kernel_x, dtype=y_pred.dtype)

    kernel_y = np.tile(kernel.T[..., None, None], (1, 1, channels, 1))
    kernel_y = tf.constant(kernel_y, dtype=y_pred.dtype)

    y_pred_x, y_pred_y = _gauss_gradient(y_pred, size, kernel_x, kernel_y)
    y_true_x, y_true_y = _gauss_gradient(y_true, size, kernel_x, kernel_y)

    pred_amp = tf.sqrt(y_pred_x ** 2 + y_pred_y ** 2)
    true_amp = tf.sqrt(y_true_x ** 2 + y_true_y ** 2)

    result = tf.math.squared_difference(pred_amp, true_amp)

    if sample_weight is not None:
        result *= sample_weight

    axis_hwc = list(range(1, result.shape.ndims))
    result = tf.reduce_sum(result, axis=axis_hwc)

    return result
