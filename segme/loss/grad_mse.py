import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from ..metric.grad import _togray, _gauss_filter, _gauss_gradient


@tf.keras.utils.register_keras_serializable(package='SegMe')
class GradientMeanSquaredError(LossFunctionWrapper):
    """ Proposed in: 'Learning-based Sampling for Natural Image Matting'

    Implements Equation [7] in https://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_Learning-Based_Sampling_for_Natural_Image_Matting_CVPR_2019_paper.pdf
    """

    def __init__(self, sigma=1.4, reduction=tf.keras.losses.Reduction.AUTO, name='gradient_mean_squared_error'):
        super().__init__(gradient_mean_squared_error, reduction=reduction, name=name, sigma=sigma)


def gradient_mean_squared_error(y_true, y_pred, sigma):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=y_pred.dtype.base_dtype)

        y_pred = _togray(y_pred)
        y_true = _togray(y_true)

        kernel, size = _gauss_filter(sigma)
        kernel_x = tf.constant(kernel[..., None, None], dtype=y_pred.dtype)
        kernel_y = tf.constant(kernel.T[..., None, None], dtype=y_pred.dtype)

        y_pred_x, y_pred_y = _gauss_gradient(y_pred, size, kernel_x, kernel_y)
        y_true_x, y_true_y = _gauss_gradient(y_true, size, kernel_x, kernel_y)

        pred_amp = tf.sqrt(y_pred_x ** 2 + y_pred_y ** 2 + epsilon)
        true_amp = tf.sqrt(y_true_x ** 2 + y_true_y ** 2 + epsilon)

        loss = tf.math.squared_difference(pred_amp, true_amp)
        loss = tf.reduce_mean(loss, axis=-1)

        return loss
