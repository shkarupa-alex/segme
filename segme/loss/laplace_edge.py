import numpy as np
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class LaplaceEdgeSigmoidCrossEntropy(LossFunctionWrapper):
    """ Proposed in: 'Pyramid Feature Attention Network for Saliency detection (2019)'

    Implements Equation [10] in https://arxiv.org/pdf/1903.00179.pdf
    Compute edge loss with Laplace operator
    """

    def __init__(
            self, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
            name='laplace_edge_sigmoid_cross_entropy'):
        super().__init__(
            laplace_edge_sigmoid_cross_entropy, reduction=reduction, name=name, from_logits=from_logits)


def laplace(probs):
    laplace = np.reshape([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [3, 3, 1, 1])
    laplace = tf.constant(laplace, probs.dtype)
    edge = tf.nn.conv2d(probs, laplace, strides=[1] * 4, padding='SAME')
    edge = tf.nn.relu(tf.tanh(edge))

    return edge


@tf.keras.utils.register_keras_serializable(package='SegMe')
def laplace_edge_sigmoid_cross_entropy(y_true, y_pred, from_logits=False):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        if from_logits:
            y_pred = tf.sigmoid(y_pred)

        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=y_pred.dtype)

        y_true_edge = laplace(y_true)
        y_pred_edge = laplace(y_pred)
        y_pred_edge = tf.clip_by_value(y_pred_edge, epsilon, 1. - epsilon)
        y_pred_edge = tf.math.log(y_pred_edge / (1. - y_pred_edge))

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_edge, logits=y_pred_edge)

        return tf.reduce_mean(loss, axis=-1)
