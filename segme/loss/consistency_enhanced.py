import tensorflow as tf
from keras import backend
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss, to_probs, to_1hot
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class ConsistencyEnhancedLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Multi-scale Interactive Network for Salient Object Detection'

    Implements Equation [9] in https://arxiv.org/pdf/2007.09062.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='consistency_enhanced_loss'):
        super().__init__(consistency_enhanced_loss, reduction=reduction, name=name, from_logits=from_logits)


def consistency_enhanced_loss(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')
    y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
    y_true, y_pred = to_1hot(y_true, y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    intersection = y_true * y_pred
    numerator0 = weighted_loss(y_pred - intersection, sample_weight)
    numerator1 = weighted_loss(y_true - intersection, sample_weight)
    denominator0 = weighted_loss(y_pred, sample_weight)
    denominator1 = weighted_loss(y_true, sample_weight)

    epsilon = backend.epsilon()
    loss = numerator0 + numerator1
    loss /= (denominator0 + denominator1 + epsilon)

    return loss
