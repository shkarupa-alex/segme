import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy, iou, mae


@register_keras_serializable(package='SegMe>Loss')
class AdaptivePixelIntensityLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'TRACER: Extreme Attention Guided Salient Object Tracing Network'

    Implements Equation (12) from https://arxiv.org/pdf/2112.07380.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='adaptive_pixel_intensity_loss'):
        super().__init__(adaptive_pixel_intensity_loss, reduction=reduction, name=name, from_logits=from_logits)


def adaptive_pixel_intensity_loss(y_true, y_pred, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=4, channel='sparse')

    y_true_1h = tf.one_hot(y_true[..., 0], max(2, y_pred.shape[-1]), dtype=y_pred.dtype)

    min_shape = tf.reduce_min(tf.shape(y_true)[1:3])
    assert_shape = tf.assert_greater(min_shape, 30)
    with tf.control_dependencies([assert_shape]):
        omega = sum([
            tf.abs(tf.nn.avg_pool2d(y_true_1h, ksize=k, strides=1, padding='SAME') - y_true_1h)
            for k in [3, 15, 31]
        ]) * y_true_1h * .5 + 1.  # TODO: reduce max by channel?
        omega = tf.math.divide_no_nan(omega, tf.reduce_mean(omega, axis=[1, 2], keepdims=True))
        omega = omega if sample_weight is None else omega * sample_weight
        omega = tf.stop_gradient(omega)

    weight = omega if sample_weight is None else omega * sample_weight

    # Skipped omega normalization from original paper
    ace = crossentropy(y_true, y_pred, weight, from_logits)
    aiou = iou(y_true, y_pred, weight, from_logits=from_logits, square=False, smooth=1., dice=False)
    amae = mae(y_true, y_pred, weight, from_logits=from_logits)

    loss = ace + aiou + amae

    return tf.reduce_mean(loss, axis=-1)
