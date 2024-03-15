import tensorflow as tf
from tf_keras import backend
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, iou
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class GeneralizedDiceLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations'

    Implements Equations from https://arxiv.org/pdf/1707.03237v3.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='generalized_dice_loss'):
        super().__init__(generalized_dice_loss, reduction=reduction, name=name, from_logits=from_logits)


def generalized_dice_loss(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')

    y_true_1h = tf.one_hot(tf.squeeze(y_true, -1), max(2, y_pred.shape[-1]), dtype=y_pred.dtype)
    weight = tf.reduce_mean(y_true_1h, axis=[0, 1, 2], keepdims=True) ** 2
    weight = tf.reduce_max(weight * y_true_1h, axis=-1, keepdims=True) + backend.epsilon()
    weight = 1. / weight

    sample_weight = weight if sample_weight is None else sample_weight * weight
    sample_weight = tf.stop_gradient(sample_weight)

    loss = iou(y_true, y_pred, sample_weight, from_logits=from_logits, smooth=1., dice=True)

    return loss
