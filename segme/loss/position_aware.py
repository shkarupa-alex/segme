import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy, iou


@register_keras_serializable(package='SegMe>Loss')
class PixelPositionAwareLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'F3Net: Fusion, Feedback and Focus for Salient Object Detection'

    Implements Equation [6] in https://arxiv.org/pdf/1911.11445.pdf (weighted BCE + weighted IoU)
    """

    def __init__(
            self, from_logits=False, gamma=5, ksize=31, reduction=Reduction.AUTO,
            name='pixel_position_aware_loss'):
        super().__init__(
            pixel_position_aware_loss, reduction=reduction, name=name, from_logits=from_logits,
            gamma=gamma, ksize=ksize)


def pixel_position_aware_loss(y_true, y_pred, from_logits, gamma, ksize):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=4, channel='sparse')

    y_true_1h = tf.one_hot(y_true[..., 0], max(2, y_pred.shape[-1]), dtype=y_pred.dtype)

    min_shape = tf.reduce_min(tf.shape(y_true)[1:3])
    assert_shape = tf.assert_greater(min_shape, ksize - 1)
    with tf.control_dependencies([assert_shape]):
        weight = 1 + gamma * tf.abs(tf.nn.avg_pool2d(y_true_1h, ksize=ksize, strides=1, padding='SAME') - y_true_1h)
        weight = tf.stop_gradient(weight)

    sample_weight = weight if sample_weight is None else sample_weight * weight

    wce = crossentropy(y_true, y_pred, sample_weight, from_logits)
    wiou = iou(y_true, y_pred, sample_weight, from_logits=from_logits, square=False, smooth=1., dice=False)

    loss = wce + wiou

    return tf.reduce_mean(loss, axis=-1)
