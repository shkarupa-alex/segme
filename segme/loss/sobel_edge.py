import numpy as np
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class SobelEdgeLoss(LossFunctionWrapper):
    """ Proposed in: 'CascadePSP: Toward Class-Agnostic and Very High-Resolution
    Segmentation via Global and Local Refinement'

    Implements Equation from https://arxiv.org/pdf/2005.02551.pdf
    Compute edge loss with Sobel operator
    """

    def __init__(
            self, classes=1, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
            name='sobel_edge_loss'):
        super().__init__(
            sobel_edge_loss, classes=classes, reduction=reduction, name=name, from_logits=from_logits)


def sobel(probs, classes, epsilon):
    sobel_x = np.reshape([[1, 0, -1], [2, 0, -2], [1, 0, -1]], [3, 3, 1, 1]) / 4
    sobel_x = np.tile(sobel_x, [1, 1, classes, 1])
    sobel_x = tf.constant(sobel_x, probs.dtype)

    sobel_y = np.reshape([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [3, 3, 1, 1]) / 4
    sobel_y = np.tile(sobel_y, [1, 1, classes, 1])
    sobel_y = tf.constant(sobel_y, probs.dtype)

    pooled = tf.nn.avg_pool2d(probs, 3, strides=1, padding='SAME')
    grad_x = tf.nn.depthwise_conv2d(pooled, sobel_x, strides=[1] * 4, padding='SAME')
    grad_y = tf.nn.depthwise_conv2d(pooled, sobel_y, strides=[1] * 4, padding='SAME')

    edge = tf.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)

    return edge


def sobel_edge_loss(y_true, y_pred, classes, from_logits):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=y_pred.dtype.base_dtype)

        if from_logits:
            y_pred = tf.sigmoid(y_pred)

        y_true_edge = sobel(y_true, classes, epsilon)
        y_pred_edge = sobel(y_pred, classes, epsilon)

        loss = tf.keras.losses.mean_absolute_error(y_true=y_true_edge, y_pred=y_pred_edge)

        return loss
