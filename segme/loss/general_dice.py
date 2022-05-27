import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input, iou


@register_keras_serializable(package='SegMe')
class GeneralizedDiceLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations'

    Implements Equations from https://arxiv.org/pdf/1707.03237v3.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='generalized_dice_loss'):
        super().__init__(generalized_dice_loss, reduction=reduction, name=name, from_logits=from_logits)


def generalized_dice_loss(y_true, y_pred, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=4, channel='sparse')

    y_true_1h = tf.one_hot(y_true[..., 0], max(2, y_pred.shape[-1]), dtype=y_pred.dtype)
    weight = (1. if sample_weight is None else sample_weight) / (
            tf.reduce_sum(y_true_1h, axis=[1, 2], keepdims=True) ** 2 + 1.)
    weight = tf.stop_gradient(weight)

    loss = iou(y_true, y_pred, weight, from_logits=from_logits, square=False, smooth=1., dice=True)

    return tf.reduce_sum(loss, axis=-1)
