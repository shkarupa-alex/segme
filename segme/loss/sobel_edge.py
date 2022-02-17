import numpy as np
import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input, to_probs, to_1hot


@register_keras_serializable(package='SegMe')
class SobelEdgeLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'CascadePSP: Toward Class-Agnostic and Very High-Resolution
    Segmentation via Global and Local Refinement'

    Implements Equation from https://arxiv.org/pdf/2005.02551.pdf
    Compute edge loss with Sobel operator
    """

    def __init__(
            self, from_logits=False, reduction=Reduction.AUTO,
            name='sobel_edge_loss'):
        super().__init__(
            sobel_edge_loss, reduction=reduction, name=name, from_logits=from_logits)


def sobel(probs):
    classes = probs.shape[-1]
    sobel_x = np.reshape([[1, 0, -1], [2, 0, -2], [1, 0, -1]], [3, 3, 1, 1]) / 4
    sobel_x = np.tile(sobel_x, [1, 1, classes, 1])
    sobel_x = tf.cast(sobel_x, probs.dtype)

    sobel_y = np.reshape([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [3, 3, 1, 1]) / 4
    sobel_y = np.tile(sobel_y, [1, 1, classes, 1])
    sobel_y = tf.cast(sobel_y, probs.dtype)

    pooled = tf.nn.avg_pool2d(probs, 3, strides=1, padding='SAME')
    grad_x = tf.nn.depthwise_conv2d(pooled, sobel_x, strides=[1] * 4, padding='SAME')
    grad_y = tf.nn.depthwise_conv2d(pooled, sobel_y, strides=[1] * 4, padding='SAME')

    edge = tf.sqrt(grad_x ** 2 + grad_y ** 2 + backend.epsilon())

    return edge


def sobel_edge_loss(y_true, y_pred, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=4, channel='sparse')
    y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
    y_true, y_pred = to_1hot(y_true, y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    y_true_edge = sobel(y_true)
    y_true_edge = tf.stop_gradient(y_true_edge)
    y_pred_edge = sobel(y_pred)

    loss = losses.mean_absolute_error(y_true=y_true_edge, y_pred=y_pred_edge)

    return loss
