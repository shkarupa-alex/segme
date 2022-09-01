import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, to_probs, to_1hot


@register_keras_serializable(package='SegMe>Loss')
class ConsistencyEnhancedLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'Multi-scale Interactive Network for Salient Object Detection'

    Implements Equation [9] in https://arxiv.org/pdf/2007.09062.pdf
    """

    def __init__(
            self, from_logits=False, reduction=Reduction.AUTO,
            name='consistency_enhanced_loss'):
        super().__init__(
            consistency_enhanced_loss, reduction=reduction, name=name, from_logits=from_logits)


def consistency_enhanced_loss(y_true, y_pred, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=4, channel='sparse')
    y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
    y_true, y_pred = to_1hot(y_true, y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    cel = tf.math.divide_no_nan(
        y_pred + y_true - 2 * y_true * y_pred,
        tf.reduce_sum(tf.reduce_mean(y_pred + y_true, axis=-1, keepdims=True), axis=[1, 2], keepdims=True))

    return tf.reduce_mean(cel, axis=-1)
