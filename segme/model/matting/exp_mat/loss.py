from functools import partial

from keras.src import ops

from segme.loss import GradientMeanAbsoluteError
from segme.loss import HardGradientMeanAbsoluteError
from segme.loss import LaplacianPyramidLoss
from segme.loss import MeanAbsoluteRegressionError
from segme.loss import ReflectionTransmissionExclusionLoss
from segme.loss import SmoothGradientPenalty
from segme.loss import WeightedLossFunctionWrapper


def _l1(y_true, y_pred, sample_weight):
    return MeanAbsoluteRegressionError()(
        y_true, y_pred, sample_weight=sample_weight
    )


def _lap(y_true, y_pred, sample_weight, levels, size, sigma, residual):
    return LaplacianPyramidLoss(
        levels=levels,
        size=size,
        sigma=sigma,
        residual=residual,
        weight_pooling="max",
    )(y_true, y_pred, sample_weight=sample_weight)


def lc_a(f_true, b_true, c_true, a_pred, sample_weight):
    c_pred = a_pred * f_true + (1.0 - a_pred) * b_true

    return _l1(c_true, c_pred, sample_weight)


def lc_fb(a_true, c_true, f_pred, b_pred, sample_weight):
    c_pred = a_true * f_pred + (1.0 - a_true) * b_pred

    return _l1(c_true, c_pred, sample_weight)


def lc_c(c_true, a_pred, f_pred, b_pred, sample_weight):
    c_pred = a_pred * f_pred + (1.0 - a_pred) * b_pred

    return _l1(c_true, c_pred, sample_weight)


def lexcl_fb(f_pred, b_pred, sample_weight, levels):
    return ReflectionTransmissionExclusionLoss(levels=levels)(
        f_pred, b_pred, sample_weight=sample_weight
    )


def lg_hard(a_true, a_pred, sample_weight):
    return HardGradientMeanAbsoluteError()(
        a_true, a_pred, sample_weight=sample_weight
    )


def lg_soft(a_true, a_pred, sample_weight, sigma):
    return GradientMeanAbsoluteError(sigma=sigma)(
        a_true, a_pred, sample_weight=sample_weight
    )


def lg_smooth(a_true, a_pred, sample_weight, strength):
    return SmoothGradientPenalty(strength=strength)(
        a_true, a_pred, sample_weight=sample_weight
    )


def exp_mat_loss(
    y_true,
    y_pred,
    sample_weight,
    total_scale,
    lap_scale,
    comp_scale,
    lap_levels,
    lap_size,
    lap_sigma,
    lap_residual,
    fb_scale,
    excl_scale,
    excl_unkn,
    excl_levels,
    hard_scale,
    soft_scale,
    soft_sigma,
    smooth_scale,
    smooth_strength,
):
    a_true, f_true, b_true, t_true = ops.split(y_true, [1, 4, 7], axis=-1)
    a_pred, f_pred, b_pred = ops.split(y_pred, [1, 4], axis=-1)

    if sample_weight is not None:
        raise ValueError(
            f"Expecting `sample_weight` to be `None`. "
            f"Got {type(sample_weight)}"
        )

    _l1_a = _l1(a_true, a_pred, sample_weight=None)
    _llap_a = lap_scale * _lap(
        a_true,
        a_pred,
        sample_weight=None,
        levels=lap_levels,
        size=lap_size,
        sigma=lap_sigma,
        residual=lap_residual,
    )
    c_true = a_true * f_true + (1.0 - a_true) * b_true
    _lc_a = comp_scale * lc_a(
        f_true, b_true, c_true, a_pred, sample_weight=None
    )

    loss = _l1_a + _llap_a + _lc_a

    if fb_scale > 0.0:
        _l1_f = _l1(f_true, f_pred, sample_weight=None)
        _l1_b = _l1(b_true, b_pred, sample_weight=None)
        _llap_f = lap_scale * _lap(
            f_true,
            f_pred,
            sample_weight=None,
            levels=lap_levels,
            size=lap_size,
            sigma=lap_sigma,
            residual=lap_residual,
        )
        _llap_b = lap_scale * _lap(
            b_true,
            b_pred,
            sample_weight=None,
            levels=lap_levels,
            size=lap_size,
            sigma=lap_sigma,
            residual=lap_residual,
        )
        _lc_fb = comp_scale * lc_fb(
            a_true, c_true, f_pred, b_pred, sample_weight=None
        )
        _lc_c = comp_scale * lc_c(
            c_true, a_pred, f_pred, b_pred, sample_weight=None
        )

        loss += (_l1_f + _l1_b + _llap_f + _llap_b + _lc_fb + _lc_c) * fb_scale

    if excl_scale > 0.0:
        if excl_unkn:
            excl_weight = ops.cast(
                (t_true > 0.25) & (t_true <= 0.75), t_true.dtype
            )
        else:
            excl_weight = None
        _lexcl_fb = lexcl_fb(
            f_pred, b_pred, sample_weight=excl_weight, levels=excl_levels
        )
        loss += _lexcl_fb * excl_scale

    if hard_scale > 0.0:
        _lg_hard = lg_hard(a_true, a_pred, sample_weight=None)
        loss += _lg_hard * hard_scale

    if soft_scale > 0.0:
        _lg_soft = lg_soft(a_true, a_pred, sample_weight=None, sigma=soft_sigma)
        loss += _lg_soft * soft_scale

    if smooth_scale > 0.0:
        _lg_smooth = lg_smooth(
            a_true, a_pred, sample_weight=None, strength=smooth_strength
        )
        loss += _lg_smooth * smooth_scale

    return loss * total_scale


def exp_mat_losses(
    scales=5,
    level_scale=1.5,  # [1.0 .. 2.0]
    lap_scale=1.0,  # [0.1, 1.9]
    comp_scale=1.0,  # [0.1, 1.9]
    lap_levels=5,  # [5 .. 7]
    lap_size=5,  # [5 .. crop_size // (2 ** lap_levels)]
    lap_sigma=1.056,  # [1.0 .. 2.0]
    lap_residual=False,  # [False, True]
    fb_scale=0.25,  # [0.0 .. 0.5]
    excl_scale=0.0,  # [0., fb_scale]
    excl_unkn=False,  # [False, True]
    excl_levels=3,  # [3 .. lap_levels]
    hard_scale=0.0,  # [0.0 .. 1.0]
    soft_scale=0.0,  # [0.0 .. 1.0]
    soft_sigma=1.4,  # [1.0 .. 2.0]
    smooth_scale=0.0,  # [0.0 .. 1.0]
    smooth_strength=0.01,  # [0.001, 0.01, 0.1]
):
    losses = []
    for i in range(scales):
        total_scale = level_scale ** (i - scales + 1)
        level_loss = partial(
            exp_mat_loss,
            total_scale=total_scale,
            lap_scale=lap_scale,
            comp_scale=comp_scale,
            lap_levels=lap_levels,
            lap_size=lap_size,
            lap_sigma=lap_sigma,
            lap_residual=lap_residual,
            fb_scale=fb_scale,
            excl_scale=excl_scale,
            excl_unkn=excl_unkn,
            excl_levels=excl_levels,
            hard_scale=hard_scale,
            soft_scale=soft_scale,
            soft_sigma=soft_sigma,
            smooth_scale=smooth_scale,
            smooth_strength=smooth_strength,
        )
        losses.append(WeightedLossFunctionWrapper(level_loss))

    return losses + [None] * 3
