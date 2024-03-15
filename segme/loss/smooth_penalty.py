import tensorflow as tf
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss, compute_gradient
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class SmoothGradientPenalty(WeightedLossFunctionWrapper):
    """ Proposed in: 'Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation'

    Implements Equation [4] in https://arxiv.org/pdf/2201.06889.pdf
    """

    def __init__(self, strength=0.01, reduction=Reduction.AUTO, name='smooth_gradient_penalty'):
        super().__init__(smooth_gradient_penalty, reduction=reduction, name=name, strength=strength)


def smooth_gradient_penalty(y_true, y_pred, sample_weight, strength):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    g_pred_x = compute_gradient(y_pred, 1, 'sub')
    g_pred_y = compute_gradient(y_pred, 2, 'sub')

    g_weight_x, g_weight_y = None, None
    if sample_weight is not None:
        g_weight_x = compute_gradient(sample_weight, 1, 'min')
        g_weight_y = compute_gradient(sample_weight, 2, 'min')

    loss = weighted_loss(tf.abs(g_pred_x), g_weight_x) + \
           weighted_loss(tf.abs(g_pred_y), g_weight_y)
    loss *= strength

    return loss
