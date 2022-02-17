import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class ReflectionTransmissionExclusionLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Single Image Reflection Removal with Perceptual Losses'

    Implements Equation [5] in https://arxiv.org/pdf/1806.05376.pdf
    """

    def __init__(
            self, levels=3, reduction=Reduction.AUTO, name='reflection_transmission_exclusion_loss'):
        super().__init__(reflection_transmission_exclusion_loss, reduction=reduction, name=name, levels=levels)


def _compute_gradient(inputs, axis, reduction):
    if 1 == axis:
        grad = inputs[:, 1:, :, :], inputs[:, :-1, :, :]
    elif 2 == axis:
        grad = inputs[:, :, 1:, :], inputs[:, :, :-1, :]
    else:
        raise ValueError('Unsupported axis: {}'.format(axis))

    if 'sub' == reduction:
        grad = grad[0] - grad[1]
    elif 'min' == reduction:
        grad = tf.minimum(grad[0], grad[1])
    else:
        raise ValueError('Unsupported reduction: {}'.format(reduction))

    return grad


def _exclusion_level(r_pred, t_pred, axis, sample_weight):
    grad_r = _compute_gradient(r_pred, axis, 'sub')
    grad_t = _compute_gradient(t_pred, axis, 'sub')
    grad_w = None if sample_weight is None else _compute_gradient(sample_weight, axis, 'min')

    alpha = 2. * tf.math.divide_no_nan(
        tf.reduce_mean(tf.abs(grad_r)),
        tf.reduce_mean(tf.abs(grad_t)))
    grad_rs = tf.nn.sigmoid(grad_r) * 2. - 1.
    grad_ts = tf.nn.sigmoid(grad_t * alpha) * 2. - 1.

    axis_hwc = list(range(1, r_pred.shape.ndims))
    loss = (grad_rs ** 2) * (grad_ts ** 2)
    if grad_w is not None:
        loss *= (grad_w ** 4)
    loss = tf.reduce_mean(loss, axis=axis_hwc) ** 0.25

    return loss


def _down_sample(reflections, transmissions, weights):
    height_width = tf.shape(reflections)[1:3]
    hpad, wpad = tf.unstack(height_width % 2)
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]

    reflections = tf.pad(reflections, paddings, 'REFLECT')
    transmissions = tf.pad(transmissions, paddings, 'REFLECT')
    if weights is not None:
        weights = tf.pad(weights, paddings, 'REFLECT')

    reflections = tf.nn.avg_pool(reflections, ksize=2, strides=2, padding='SAME')
    transmissions = tf.nn.avg_pool(transmissions, ksize=2, strides=2, padding='SAME')
    if weights is not None:
        weights = tf.nn.avg_pool(weights, ksize=2, strides=2, padding='SAME')
        weights = tf.stop_gradient(weights)

    return reflections, transmissions, weights


def reflection_transmission_exclusion_loss(r_pred, t_pred, sample_weight, levels):
    r_pred, t_pred, sample_weight = validate_input(
        r_pred, t_pred, sample_weight, dtype=None, rank=4, channel='same')

    loss = []
    for level in range(levels):
        loss.append(_exclusion_level(r_pred, t_pred, axis=1, sample_weight=sample_weight))
        loss.append(_exclusion_level(r_pred, t_pred, axis=2, sample_weight=sample_weight))
        last_level = levels - 1 == level
        if not last_level:
            assert_true_shape = tf.assert_greater(tf.reduce_min(tf.shape(r_pred)[1:3]), 2)

            with tf.control_dependencies([assert_true_shape]):
                r_pred, t_pred, sample_weight = _down_sample(r_pred, t_pred, sample_weight)
    loss = sum(loss) / (2. * levels)

    return loss
