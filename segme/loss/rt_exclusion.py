import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
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
        loss *= grad_w
    loss = tf.reduce_mean(loss, axis=axis_hwc) ** 0.25

    return loss


def _down_sample(reflections, transmissions, weights, level):
    if 0 == level:
        return reflections, transmissions, weights

    factor = 2 ** level
    height, width = tf.unstack(tf.shape(reflections)[1:3])

    hpad = (factor - height % factor) % factor
    wpad = (factor - width % factor) % factor
    pads = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    reflections = tf.pad(reflections, pads, 'REFLECT')
    transmissions = tf.pad(transmissions, pads, 'REFLECT')
    if weights is not None:
        weights = tf.pad(weights, pads, 'REFLECT')

    size = ((height + hpad) // factor, (width + wpad) // factor)
    reflections = tf.image.resize(reflections, size)
    transmissions = tf.image.resize(transmissions, size)
    if weights is not None:
        weights = tf.image.resize(weights, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return reflections, transmissions, weights


def reflection_transmission_exclusion_loss(r_pred, t_pred, sample_weight, levels):
    r_pred = tf.convert_to_tensor(r_pred)
    t_pred = tf.cast(t_pred, dtype=r_pred.dtype)

    assert_r_rank = tf.assert_rank(r_pred, 4)
    assert_t_rank = tf.assert_rank(t_pred, 4)

    with tf.control_dependencies([assert_r_rank, assert_t_rank]):
        loss = []
        for level in range(levels):
            r_pred_down, t_pred_down, sample_weight_down = _down_sample(r_pred, t_pred, sample_weight, level)
            loss.append(_exclusion_level(r_pred_down, t_pred_down, axis=1, sample_weight=sample_weight_down))
            loss.append(_exclusion_level(r_pred_down, t_pred_down, axis=2, sample_weight=sample_weight_down))
        loss = sum(loss) / (2. * levels)

        return loss
