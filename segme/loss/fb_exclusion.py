import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class ForegroundBackgroundExclusionLoss(LossFunctionWrapper):
    """ Proposed in: 'Single Image Reflection Removal with Perceptual Losses'

    Implements Equation [5] in https://arxiv.org/pdf/1806.05376.pdf
    """

    def __init__(
            self, levels=3, reduction=tf.keras.losses.Reduction.AUTO, name='foreground_background_exclusion_loss'):
        super().__init__(foreground_background_exclusion_loss, reduction=reduction, name=name, levels=levels)


def _foreground_background_exclusion_loss(f_pred, b_pred, axis):
    if 1 == axis:
        grad_f = f_pred[:, 1:, :, :] - f_pred[:, :-1, :, :]
        grad_b = b_pred[:, 1:, :, :] - b_pred[:, :-1, :, :]
    elif 2 == axis:
        grad_f = f_pred[:, :, 1:, :] - f_pred[:, :, :-1, :]
        grad_b = b_pred[:, :, 1:, :] - b_pred[:, :, :-1, :]
    else:
        raise ValueError('Unsupported axis: {}'.format(axis))

    axis_hwc = list(range(1, f_pred.shape.ndims))
    alpha = 2. * tf.math.divide_no_nan(
        tf.reduce_mean(tf.abs(grad_f), axis=axis_hwc, keepdims=True),
        tf.reduce_mean(tf.abs(grad_b), axis=axis_hwc, keepdims=True))
    grad_fs = tf.nn.sigmoid(grad_f) * 2. - 1.
    grad_bs = tf.nn.sigmoid(grad_b * alpha) * 2. - 1.

    loss = tf.reduce_mean(tf.multiply(grad_fs ** 2, grad_bs ** 2), axis=axis_hwc) ** 0.25

    return loss


@tf.keras.utils.register_keras_serializable(package='SegMe')
def foreground_background_exclusion_loss(f_pred, b_pred, levels=3):
    assert_f_rank = tf.assert_rank(f_pred, 4)
    assert_b_rank = tf.assert_rank(b_pred, 4)

    with tf.control_dependencies([assert_f_rank, assert_b_rank]):
        f_pred = tf.convert_to_tensor(f_pred)
        b_pred = tf.cast(b_pred, dtype=f_pred.dtype)

        loss = []
        for level in range(levels):
            if level > 0:
                f_pred = tf.nn.avg_pool(f_pred, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                b_pred = tf.nn.avg_pool(b_pred, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            loss.append(_foreground_background_exclusion_loss(f_pred, b_pred, axis=1))
            loss.append(_foreground_background_exclusion_loss(f_pred, b_pred, axis=2))

        loss = sum(loss) / (2. * levels)

        return loss
