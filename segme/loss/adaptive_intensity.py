import tensorflow as tf
from keras.saving.object_registration import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy, iou, mae
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class AdaptivePixelIntensityLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'TRACER: Extreme Attention Guided Salient Object Tracing Network'

    Implements Equation (12) from https://arxiv.org/pdf/2112.07380.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='adaptive_pixel_intensity_loss'):
        super().__init__(adaptive_pixel_intensity_loss, reduction=reduction, name=name, from_logits=from_logits)


def adaptive_pixel_intensity_loss(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')

    min_shape = tf.reduce_min(tf.shape(y_true)[1:3])
    assert_shape = tf.assert_greater(min_shape, 30)
    with tf.control_dependencies([assert_shape]):
        y_true_1h = tf.one_hot(y_true[..., 0], max(2, y_pred.shape[-1]), dtype=y_pred.dtype)
        omega = sum([
            tf.abs(tf.nn.avg_pool2d(y_true_1h, ksize=k, strides=1, padding='SAME') - y_true_1h)
            for k in [3, 15, 31]]) * y_true_1h * .5 + 1.
        omega = tf.reduce_max(omega, axis=-1, keepdims=True)

    sample_weight = omega if sample_weight is None else omega * sample_weight
    sample_weight = tf.stop_gradient(sample_weight)

    omega_mean = tf.reduce_mean(omega, axis=[1, 2, 3])
    omega_mean = tf.stop_gradient(omega_mean)

    ace = crossentropy(y_true, y_pred, sample_weight, from_logits) / (omega_mean + 0.5)
    aiou = iou(y_true, y_pred, sample_weight, from_logits=from_logits)
    amae = mae(y_true, y_pred, sample_weight, from_logits=from_logits) / (omega_mean - 0.5)  # -1 will produce NaNs

    loss = ace + aiou + amae

    return loss
