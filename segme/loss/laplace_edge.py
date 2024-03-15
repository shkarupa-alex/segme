import numpy as np
import tensorflow as tf
from tf_keras import backend
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss, to_probs, to_1hot
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class LaplaceEdgeCrossEntropy(WeightedLossFunctionWrapper):
    """ Proposed in: 'Pyramid Feature Attention Network for Saliency detection (2019)'

    Implements Equation [10] in https://arxiv.org/pdf/1903.00179.pdf
    Compute edge loss with Laplace operator
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='laplace_edge_cross_entropy'):
        super().__init__(laplace_edge_cross_entropy, reduction=reduction, name=name, from_logits=from_logits)


def laplace(probs, kernel):
    edge = tf.pad(probs, [(0, 0), (1, 1), (1, 1), (0, 0)], mode='SYMMETRIC')
    edge = tf.nn.depthwise_conv2d(edge, kernel, strides=[1] * 4, padding='VALID')
    edge = tf.nn.relu(tf.tanh(edge))

    return edge


def laplace_edge_cross_entropy(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')
    y_prob = to_probs(y_pred, from_logits, force_sigmoid=True)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob)
    y_true_1h = tf.cast(y_true_1h, y_prob.dtype)

    kernel = np.reshape([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [3, 3, 1, 1])
    kernel = np.tile(kernel, [1, 1, y_true_1h.shape[-1], 1])
    kernel = tf.cast(kernel, y_prob.dtype)
    kernel = tf.stop_gradient(kernel)

    y_true_edge = laplace(y_true_1h, kernel)[..., None]
    y_true_edge = tf.stop_gradient(y_true_edge)
    y_pred_edge = laplace(y_prob_1h, kernel)[..., None]

    loss = backend.binary_crossentropy(y_true_edge, y_pred_edge, from_logits=False)
    loss = tf.reduce_mean(loss, axis=-1)
    loss = weighted_loss(loss, sample_weight)

    return loss
