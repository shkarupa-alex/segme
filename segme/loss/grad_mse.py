import numpy as np
import tensorflow as tf
from keras import backend
from keras.saving.object_registration import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss
from segme.metric.matting.grad import _togray, _gauss_filter, _gauss_gradient
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class GradientMeanSquaredError(WeightedLossFunctionWrapper):
    """ Proposed in: 'Learning-based Sampling for Natural Image Matting'

    Implements Equation [7] in https://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_Learning-Based_Sampling_for_Natural_Image_Matting_CVPR_2019_paper.pdf
    """

    def __init__(self, sigma=1.4, reduction=Reduction.AUTO, name='gradient_mean_squared_error'):
        super().__init__(gradient_mean_squared_error, reduction=reduction, name=name, sigma=sigma)


def gradient_mean_squared_error(y_true, y_pred, sample_weight, sigma):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    y_pred = _togray(y_pred)
    y_true = _togray(y_true)

    channels = y_pred.shape[-1]
    kernel0, kernel1, size = _gauss_filter(sigma)
    kernel0 = np.tile(kernel0[..., None, None], (1, 1, channels, 1))
    kernel1 = np.tile(kernel1[..., None, None], (1, 1, channels, 1))
    kernel0 = kernel0.astype(y_pred.dtype.as_numpy_dtype)
    kernel1 = kernel1.astype(y_pred.dtype.as_numpy_dtype)

    kernel_x = tf.cast(kernel0, y_pred.dtype), tf.cast(kernel1, y_pred.dtype)
    kernel_y = kernel0.transpose([1, 0, 2, 3]), kernel1.transpose([1, 0, 2, 3])
    kernel_y = tf.cast(kernel_y[0], y_pred.dtype), tf.cast(kernel_y[1], y_pred.dtype)

    y_pred_x, y_pred_y = _gauss_gradient(y_pred, size, kernel_x, kernel_y)
    y_true_x, y_true_y = _gauss_gradient(y_true, size, kernel_x, kernel_y)

    pred_amp = tf.sqrt(y_pred_x ** 2 + y_pred_y ** 2 + backend.epsilon())
    true_amp = tf.sqrt(y_true_x ** 2 + y_true_y ** 2 + backend.epsilon())
    true_amp = tf.stop_gradient(true_amp)

    loss = tf.math.squared_difference(pred_amp, true_amp)
    loss = weighted_loss(loss, sample_weight)

    return loss
