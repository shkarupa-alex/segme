import tensorflow as tf
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss
from segme.loss.kl_divergence import _kl_divergence
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class ClipFoundationLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'DIME-FM : DIstilling Multimodal and Efficient Foundation Models'

    Implements Equations from https://arxiv.org/pdf/2303.18232.pdf
    """
    def __init__(
            self, scale=100., bias=None, temperature=(1., 3., 7.), weight=(0.68, 0.22, 0.10), reduction=Reduction.AUTO,
            name='clip_foundation_loss'):
        super().__init__(
            clip_foundation_loss, reduction=reduction, name=name, scale=scale, bias=bias, temperature=temperature,
            weight=weight)


def clip_foundation_loss(y_true, y_pred, sample_weight, scale, bias, temperature, weight):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel=None)

    if y_pred.shape[-1] * 2 != y_true.shape[-1]:
        raise ValueError('Labels channel size must be twice larger then predictions one.')
    y_true_vision, y_true_text = tf.split(y_true, 2, axis=-1)

    vl_temp, pvl_temp, udist_temp = temperature
    vl_weight, pvl_weight, udist_weight = weight

    # vl loss
    tv_tt, tt_tv = _compute_similarity(y_true_vision, y_true_text, scale, bias, True)
    sv_tt, tt_sv = _compute_similarity(y_pred, y_true_text, scale, bias, True)
    loss = vl_weight * (_kl_divergence(tv_tt, sv_tt, vl_temp) + _kl_divergence(tt_tv, tt_sv, vl_temp))

    if pvl_weight or udist_weight:
        tv_tv = _compute_similarity(y_true_vision, y_true_vision, scale, bias, False)
    else:
        tv_tv = None

    # pvl loss
    if pvl_weight:
        sv_tv = _compute_similarity(y_pred, y_true_vision, scale, bias, False)
        loss += pvl_weight * 2 * _kl_divergence(tv_tv, sv_tv, pvl_temp)

    # udist loss
    if udist_weight:
        sv_sv = _compute_similarity(y_pred, y_pred, scale, bias, False)
        loss += udist_weight * 2 * _kl_divergence(tv_tv, sv_sv, udist_temp)

    loss = tf.reshape(loss, tf.shape(y_true)[:-1])[..., None]

    return weighted_loss(loss, sample_weight)


def _compute_similarity(a, b, scale, bias, symmetric):
    a = tf.reshape(a, [-1, a.shape[-1]])
    b = tf.reshape(b, [-1, b.shape[-1]])

    a /= tf.norm(a, axis=-1, keepdims=True)
    b /= tf.norm(b, axis=-1, keepdims=True)

    c = tf.matmul(a * scale, b, transpose_b=True)
    if bias:
        c += bias

    if not symmetric:
        return c

    d = tf.matmul(b * scale, a, transpose_b=True)
    if bias:
        d += bias

    return c, d
