import tensorflow as tf
from keras.saving.object_registration import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss, compute_gradient
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class HardGradientMeanAbsoluteError(WeightedLossFunctionWrapper):
    """ Proposed in: 'Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation'

    Implements Equation [4] in https://arxiv.org/pdf/2201.06889.pdf
    """

    def __init__(
            self, weight=1., smooth=0.01, reduction=Reduction.AUTO, name='hard_gradient_mean_absolute_error'):
        super().__init__(
            hard_gradient_mean_absolute_error, reduction=reduction, name=name, weight=weight, smooth=smooth)


def hard_gradient_mean_absolute_error(y_true, y_pred, sample_weight, weight, smooth):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    g_true_x = compute_gradient(y_true, 1, 'sub')
    g_true_y = compute_gradient(y_true, 2, 'sub')

    g_pred_x = compute_gradient(y_pred, 1, 'sub')
    g_pred_y = compute_gradient(y_pred, 2, 'sub')

    g_weight_x, g_weight_y = None, None
    if sample_weight is not None:
        g_weight_x = compute_gradient(sample_weight, 1, 'min')
        g_weight_y = compute_gradient(sample_weight, 2, 'min')

    g_loss_x = weighted_loss(tf.abs(g_true_x - g_pred_x), g_weight_x) * weight
    g_loss_y = weighted_loss(tf.abs(g_true_y - g_pred_y), g_weight_y) * weight
    loss = [g_loss_x, g_loss_y]

    if smooth > 0:
        s_loss_x = weighted_loss(tf.abs(g_pred_x), g_weight_x) * smooth
        s_loss_y = weighted_loss(tf.abs(g_pred_y), g_weight_y) * smooth
        loss.extend([s_loss_x, s_loss_y])

    loss = sum(loss)

    return loss
