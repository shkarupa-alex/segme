from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.kl_divergence import _kl_divergence
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class ClipFoundationLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'DIME-FM : DIstilling Multimodal and Efficient Foundation
    Models'

    Implements Equations from https://arxiv.org/pdf/2303.18232
    """

    def __init__(
        self,
        scale=100.0,
        bias=None,
        temperature=(1.0, 3.0, 7.0),
        weight=(0.68, 0.22, 0.10),
        reduction="sum_over_batch_size",
        name="clip_foundation_loss",
    ):
        super().__init__(
            clip_foundation_loss,
            reduction=reduction,
            name=name,
            scale=scale,
            bias=bias,
            temperature=temperature,
            weight=weight,
        )


def clip_foundation_loss(
    y_true, y_pred, sample_weight, scale, bias, temperature, weight
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel=None
    )

    if y_pred.shape[-1] * 2 != y_true.shape[-1]:
        raise ValueError(
            "Labels channel size must be twice larger then predictions one."
        )
    y_true_vision, y_true_text = ops.split(y_true, 2, axis=-1)

    vl_temp, pvl_temp, udist_temp = temperature
    vl_weight, pvl_weight, udist_weight = weight

    # vl loss
    tv_tt, tt_tv = _compute_similarity(
        y_true_vision, y_true_text, scale, bias, True
    )
    sv_tt, tt_sv = _compute_similarity(y_pred, y_true_text, scale, bias, True)
    loss = vl_weight * (
        _kl_divergence(tv_tt, sv_tt, vl_temp)
        + _kl_divergence(tt_tv, tt_sv, vl_temp)
    )

    if pvl_weight or udist_weight:
        tv_tv = _compute_similarity(
            y_true_vision, y_true_vision, scale, bias, False
        )
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

    loss = ops.reshape(loss, ops.shape(y_true)[:-1])[..., None]

    return weighted_loss(loss, sample_weight)


def _compute_similarity(a, b, scale, bias, symmetric):
    a = ops.reshape(a, [-1, a.shape[-1]])
    b = ops.reshape(b, [-1, b.shape[-1]])

    a /= ops.norm(a, axis=-1, keepdims=True)
    b /= ops.norm(b, axis=-1, keepdims=True)

    c = ops.matmul(a * scale, ops.moveaxis(b, -1, -2))
    if bias:
        c += bias

    if not symmetric:
        return c

    d = ops.matmul(b * scale, ops.moveaxis(a, -1, -2))
    if bias:
        d += bias

    return c, d
