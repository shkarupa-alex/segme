import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper
from segme.metric.matting.grad import _gauss_filter
from segme.metric.matting.grad import _gauss_gradient
from segme.metric.matting.grad import _togray


@register_keras_serializable(package="SegMe>Loss")
class GradientMeanSquaredError(WeightedLossFunctionWrapper):
    """Proposed in: 'Learning-based Sampling for Natural Image Matting'

    Implements Equation [7] in https://openaccess.thecvf.com/
    content_CVPR_2019/papers/
    Tang_Learning-Based_Sampling_for_Natural_Image_Matting_CVPR_2019_paper.pdf
    """

    def __init__(
        self,
        sigma=1.4,
        reduction="sum_over_batch_size",
        name="gradient_mean_squared_error",
    ):
        super().__init__(
            gradient_mean_squared_error,
            reduction=reduction,
            name=name,
            sigma=sigma,
        )


def gradient_mean_squared_error(y_true, y_pred, sample_weight, sigma):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel="same"
    )

    y_pred = _togray(y_pred)
    y_true = _togray(y_true)

    channels = y_pred.shape[-1]
    kernel0, kernel1, size = _gauss_filter(sigma)
    kernel0 = np.tile(kernel0[..., None, None], (1, 1, channels, 1))
    kernel1 = np.tile(kernel1[..., None, None], (1, 1, channels, 1))

    dtype = backend.standardize_dtype(y_pred.dtype)
    kernel0 = kernel0.astype(dtype)
    kernel1 = kernel1.astype(dtype)

    kernel_x = ops.cast(kernel0, y_pred.dtype), ops.cast(kernel1, y_pred.dtype)
    kernel_y = kernel0.transpose([1, 0, 2, 3]), kernel1.transpose([1, 0, 2, 3])
    kernel_y = ops.cast(kernel_y[0], y_pred.dtype), ops.cast(
        kernel_y[1], y_pred.dtype
    )

    y_pred_x, y_pred_y = _gauss_gradient(y_pred, size, kernel_x, kernel_y)
    y_true_x, y_true_y = _gauss_gradient(y_true, size, kernel_x, kernel_y)

    pred_amp = ops.sqrt(y_pred_x**2 + y_pred_y**2 + backend.epsilon())
    true_amp = ops.sqrt(y_true_x**2 + y_true_y**2 + backend.epsilon())
    true_amp = ops.stop_gradient(true_amp)

    loss = ops.square(pred_amp - true_amp)
    loss = weighted_loss(loss, sample_weight)

    return loss
