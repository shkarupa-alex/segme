import numpy as np
import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input, to_probs, to_1hot, crossentropy


@register_keras_serializable(package='SegMe')
class LaplaceEdgeCrossEntropy(losses.LossFunctionWrapper):
    """ Proposed in: 'Pyramid Feature Attention Network for Saliency detection (2019)'

    Implements Equation [10] in https://arxiv.org/pdf/1903.00179.pdf
    Compute edge loss with Laplace operator
    """

    def __init__(
            self, from_logits=False, reduction=Reduction.AUTO,
            name='laplace_edge_cross_entropy'):
        super().__init__(
            laplace_edge_cross_entropy, reduction=reduction, name=name, from_logits=from_logits)


def laplace(probs):
    classes = probs.shape[-1]
    laplace = np.reshape([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [3, 3, 1, 1])
    laplace = np.tile(laplace, [1, 1, classes, 1])
    laplace = tf.cast(laplace, probs.dtype)
    edge = tf.nn.conv2d(probs, laplace, strides=[1] * 4, padding='SAME')
    edge = tf.nn.relu(tf.tanh(edge))

    return edge


def laplace_edge_cross_entropy(y_true, y_pred, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=4, channel='sparse')
    y_prob = to_probs(y_pred, from_logits, force_sigmoid=True)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob)
    y_true_1h = tf.cast(y_true_1h, y_prob_1h.dtype)

    y_true_edge = laplace(y_true_1h)
    y_true_edge = tf.stop_gradient(y_true_edge)

    y_pred_edge = laplace(y_prob_1h)
    y_pred_edge = tf.clip_by_value(y_pred_edge, backend.epsilon(), 1. - backend.epsilon())
    y_pred_edge = tf.math.log(y_pred_edge / (1. - y_pred_edge))

    loss = crossentropy(y_true_edge, y_pred_edge, sample_weight=None, from_logits=True)

    return tf.reduce_mean(loss, axis=-1)
