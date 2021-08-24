import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction


@register_keras_serializable(package='SegMe')
class ConsistencyEnhancedSigmoidLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'Multi-scale Interactive Network for Salient Object Detection'

    Implements Equation [9] in https://arxiv.org/pdf/2007.09062.pdf
    """

    def __init__(
            self, from_logits=False, reduction=Reduction.AUTO,
            name='consistency_enhanced_sigmoid_loss'):
        super().__init__(
            consistency_enhanced_sigmoid_loss, reduction=reduction, name=name, from_logits=from_logits)


def consistency_enhanced_sigmoid_loss(y_true, y_pred, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    cel = tf.math.divide_no_nan(y_pred + y_true - 2 * y_true * y_pred, tf.reduce_sum(y_pred + y_true))

    return tf.reduce_mean(cel, axis=-1)
