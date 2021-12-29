import tensorflow as tf
from keras import backend
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class GeneralizedDiceLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations'

    Implements Equations from https://arxiv.org/pdf/1707.03237v3.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='generalized_dice_loss'):
        super().__init__(generalized_dice_loss, reduction=reduction, name=name, from_logits=from_logits)


def generalized_dice_loss(y_true, y_pred, sample_weight, from_logits):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        epsilon = tf.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype.base_dtype)

        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        y_pred = tf.concat([y_pred, 1. - y_pred], axis=-1)
        y_true = tf.concat([y_true, 1. - y_true], axis=-1)
        y_true = tf.stop_gradient(y_true)

        weight = y_true if sample_weight is None else y_true * sample_weight
        weight = 1. / (tf.reduce_sum(weight, axis=[1, 2]) ** 2 + epsilon)
        weight = tf.stop_gradient(weight)

        intersection = y_pred * y_true if sample_weight is None else y_pred * y_true * sample_weight
        intersection = weight * tf.reduce_sum(intersection, axis=[1, 2])
        intersection = tf.reduce_sum(intersection, axis=-1)
        union = y_pred + y_true if sample_weight is None else (y_pred + y_true) * sample_weight
        union = weight * tf.reduce_sum(union, axis=[1, 2])
        union = tf.reduce_sum(union, axis=-1)
        loss = 1. - (2. * intersection + epsilon) / (union + epsilon)

        return loss
