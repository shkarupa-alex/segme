import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction


@register_keras_serializable(package='SegMe')
class NormalizedFocalSigmoidCrossEntropy(losses.LossFunctionWrapper):
    """ Proposed in: 'AdaptIS: Adaptive Instance Selection Network'

    Implements Equations (Appendix A) from https://arxiv.org/pdf/1909.07829v1.pdf
    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """

    def __init__(
            self, from_logits=False, alpha=0.25, gamma=2, reduction=Reduction.AUTO,
            name='normalized_focal_sigmoid_cross_entropy'):
        super().__init__(
            normalized_focal_sigmoid_cross_entropy, reduction=reduction, name=name, from_logits=from_logits,
            alpha=alpha, gamma=gamma)


def normalized_focal_sigmoid_cross_entropy(y_true, y_pred, alpha, gamma, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    assert_true_rank = tf.assert_greater(y_true.shape.ndims, 2)
    assert_pred_rank = tf.assert_greater(y_pred.shape.ndims, 2)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        epsilon = tf.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype)

        ce = backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
        pred_prob = y_pred if not from_logits else tf.sigmoid(y_pred)

        p_t = y_true * pred_prob + (1 - y_true) * (1. - pred_prob)

        alpha = tf.convert_to_tensor(alpha, dtype=y_pred.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1. - alpha)

        gamma = tf.convert_to_tensor(gamma, dtype=y_pred.dtype)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        axis_hw = list(range(1, y_pred.shape.ndims - 1))
        mod_mean = tf.reduce_mean(modulating_factor, axis=axis_hw, keepdims=True)
        modulating_factor /= (mod_mean + epsilon)

        total_factor = tf.stop_gradient(alpha_factor * modulating_factor)

        return tf.reduce_sum(total_factor * ce, axis=-1)
