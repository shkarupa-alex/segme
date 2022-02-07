import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class BinaryAdaptivePixelIntensityLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'TRACER: Extreme Attention Guided Salient Object Tracing Network'

    Implements Equation (12) from https://arxiv.org/pdf/2112.07380.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='binary_adaptive_pixel_intensity_loss'):
        super().__init__(binary_adaptive_pixel_intensity_loss, reduction=reduction, name=name, from_logits=from_logits)


def binary_adaptive_pixel_intensity_loss(y_true, y_pred, sample_weight, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        epsilon = tf.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype)

        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        weight = [tf.abs(tf.nn.avg_pool2d(y_true, ksize=ksize, strides=1, padding='SAME') - y_true)
                  for ksize in [3, 15, 31]]
        omega = 1 + 0.5 * sum(weight) * y_true

        bce = backend.binary_crossentropy(y_true, y_pred, from_logits=False)
        bce = bce if sample_weight is None else bce * sample_weight
        abce = tf.reduce_sum(omega * bce, axis=[1, 2]) / tf.reduce_sum(omega + 0.5, axis=[1, 2])

        intersection = y_pred * y_true * omega
        intersection = intersection if sample_weight is None else intersection * sample_weight
        intersection = tf.reduce_sum(intersection, axis=[1, 2])
        union = (y_pred + y_true) * omega
        union = union if sample_weight is None else union * sample_weight
        union = tf.reduce_sum(union, axis=[1, 2])
        aiou = 1. - (intersection + 1.) / (union - intersection + 1.)

        mae = tf.abs(y_pred - y_true)
        mae = mae if sample_weight is None else mae * sample_weight
        amae = tf.reduce_sum(omega * mae, axis=[1, 2]) / tf.reduce_sum(omega - 1. + epsilon, axis=[1, 2])

        return tf.reduce_mean(abce + aiou + amae, axis=-1)
