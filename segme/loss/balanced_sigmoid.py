import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_module


@tf.keras.utils.register_keras_serializable(package='SegMe')
class BalancedSigmoidCrossEntropy(LossFunctionWrapper):
    """ Initially proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'

    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """

    def __init__(
            self, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
            name='balanced_sigmoid_cross_entropy'):
        super().__init__(
            balanced_sigmoid_cross_entropy, reduction=reduction, name=name, from_logits=from_logits)


@tf.keras.utils.register_keras_serializable(package='SegMe')
def balanced_sigmoid_cross_entropy(y_true, y_pred, from_logits=False):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    if not from_logits:
        # When sigmoid activation function is used for output operation, we use logits from the sigmoid function
        # directly to compute loss in order to prevent collapsing zero when training.
        if isinstance(y_pred, (ops.EagerTensor, variables_module.Variable)) or y_pred.op.type != 'Sigmoid':
            print(y_pred.op)
            raise ValueError('Unable to get back logits from predictions.')
        if len(y_pred.op.inputs) != 1:
            raise ValueError('Bad sigmoid input size.')
        y_pred = y_pred.op.inputs[0]

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    beta = count_neg / (count_neg + count_pos)  # Equation [2]
    pos_weight = beta / (1. - beta)  # Equation [2] divide by 1 - beta
    cost = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1. - beta), axis=-1)  # Multiply by 1 - beta

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
