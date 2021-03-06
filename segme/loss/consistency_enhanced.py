import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class ConsistencyEnhancedSigmoidLoss(LossFunctionWrapper):
    """ Proposed in: 'Multi-scale Interactive Network for Salient Object Detection'

    Implements Equation [9] in https://arxiv.org/pdf/2007.09062.pdf
    """

    def __init__(
            self, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
            name='consistency_enhanced_sigmoid_loss'):
        super().__init__(
            consistency_enhanced_sigmoid_loss, reduction=reduction, name=name, from_logits=from_logits)


@tf.keras.utils.register_keras_serializable(package='SegMe')
def consistency_enhanced_sigmoid_loss(y_true, y_pred, from_logits=False):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=y_pred.dtype)

    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    cel = (y_pred + y_true - 2 * y_true * y_pred) / tf.reduce_sum(y_pred + y_true + epsilon)

    return tf.reduce_mean(cel, axis=-1)
