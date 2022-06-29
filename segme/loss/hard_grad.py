import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class HardGradientMeanAbsoluteError(WeightedLossFunctionWrapper):
    """ Proposed in: 'Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation'

    Implements Equation [4] in https://arxiv.org/pdf/2201.06889.pdf
    """

    def __init__(
            self, smooth=0.01, reduction=Reduction.AUTO, name='hard_gradient_mean_absolute_error'):
        super().__init__(hard_gradient_mean_absolute_error, reduction=reduction, name=name, smooth=smooth)


def _compute_gradient(inputs, axis, reduction):
    if 1 == axis:
        grad = inputs[:, 1:, :, :], inputs[:, :-1, :, :]
        pads = [[0, 0], [0, 1], [0, 0], [0, 0]]
    elif 2 == axis:
        grad = inputs[:, :, 1:, :], inputs[:, :, :-1, :]
        pads = [[0, 0], [0, 0], [0, 1], [0, 0]]
    else:
        raise ValueError('Unsupported axis: {}'.format(axis))

    if 'sub' == reduction:
        grad = grad[0] - grad[1]
    elif 'min' == reduction:
        grad = tf.minimum(grad[0], grad[1])
    else:
        raise ValueError('Unsupported reduction: {}'.format(reduction))

    grad = tf.pad(grad, pads)

    return grad


def hard_gradient_mean_absolute_error(y_true, y_pred, sample_weight, smooth):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    g_true_x = _compute_gradient(y_true, 1, 'sub')
    g_true_y = _compute_gradient(y_true, 2, 'sub')

    g_pred_x = _compute_gradient(y_pred, 1, 'sub')
    g_pred_y = _compute_gradient(y_pred, 2, 'sub')

    g_weight_x, g_weight_y = None, None
    if sample_weight is not None:
        g_weight_x = _compute_gradient(sample_weight, 1, 'min')
        g_weight_y = _compute_gradient(sample_weight, 2, 'min')

    g_loss_x = tf.abs(g_true_x - g_pred_x)
    g_loss_y = tf.abs(g_true_y - g_pred_y)
    if sample_weight is not None:
        g_loss_x *= g_weight_x
        g_loss_y *= g_weight_y
    loss = g_loss_x + g_loss_y

    if smooth > 0:
        s_loss_x = tf.abs(g_pred_x)
        s_loss_y = tf.abs(g_pred_y)
        if sample_weight is not None:
            g_loss_x *= g_weight_x
            g_loss_y *= g_weight_y
        loss += (s_loss_x + s_loss_y) * smooth

    return tf.reduce_mean(loss, axis=-1)
