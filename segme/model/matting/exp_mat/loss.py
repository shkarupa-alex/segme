from functools import partial

from keras.src import ops

from segme.loss import GradientMeanSquaredError
from segme.loss import LaplacianPyramidLoss
from segme.loss import MeanAbsoluteRegressionError
from segme.loss import ReflectionTransmissionExclusionLoss
from segme.loss import WeightedLossFunctionWrapper


def _mae(y_true, y_pred, sample_weight):
    return MeanAbsoluteRegressionError()(
        y_true, y_pred, sample_weight=sample_weight
    )


def _lap(y_true, y_pred, sample_weight, levels):
    return LaplacianPyramidLoss(levels=levels, sigma=1.056, weight_pooling="max")(
        y_true, y_pred, sample_weight=sample_weight
    )


def l1_a(a_true, a_pred, sample_weight):
    return _mae(a_true, a_pred, sample_weight)


def l1_f(f_true, f_pred, sample_weight):
    return _mae(f_true, f_pred, sample_weight)


def l1_b(b_true, b_pred, sample_weight):
    return _mae(b_true, b_pred, sample_weight)


def lc_a(f_true, b_true, c_true, a_pred, sample_weight):
    c_pred = a_pred * f_true + (1.0 - a_pred) * b_true

    return _mae(c_true, c_pred, sample_weight)


def lc_fb(a_true, c_true, f_pred, b_pred, sample_weight):
    c_pred = a_true * f_pred + (1.0 - a_true) * b_pred

    return _mae(c_true, c_pred, sample_weight)


def lexcl_fb(f_pred, b_pred, sample_weight):
    return ReflectionTransmissionExclusionLoss()(
        f_pred, b_pred, sample_weight=sample_weight
    )


def lg_a(a_true, a_pred, sample_weight):
    return GradientMeanSquaredError()(
        a_true, a_pred, sample_weight=sample_weight
    )


def llap_a(a_true, a_pred, sample_weight, level):
    return _lap(a_true, a_pred, sample_weight, level)


def llap_f(f_true, f_pred, sample_weight, level):
    return _lap(f_true, f_pred, sample_weight, level)


def llap_b(b_true, b_pred, sample_weight, level):
    return _lap(b_true, b_pred, sample_weight, level)


def exp_mat_loss(y_true, y_pred, sample_weight, scale, scales=5):
    (
        f_true,
        b_true,
        a_true,
    ) = ops.split(y_true, [3, 6], axis=-1)
    f_pred, b_pred, a_pred = ops.split(y_pred, [3, 6], axis=-1)

    a_weight, f_weight, b_weight = None, None, None
    if sample_weight is not None:
        f_weight, b_weight, a_weight = ops.split(sample_weight, 3, axis=-1)

    _l1_a = l1_a(a_true, a_pred, sample_weight=a_weight)
    _l1_f = l1_f(f_true, f_pred, sample_weight=f_weight)
    _l1_b = l1_b(b_true, b_pred, sample_weight=b_weight)

    c_true = a_true * f_true + (1.0 - a_true) * b_true
    _lc_a = lc_a(f_true, b_true, c_true, a_pred, sample_weight=None)
    _lc_fb = lc_fb(a_true, c_true, f_pred, b_pred, sample_weight=None)
    # TODO: lc_afb

    _llap_a = llap_a(a_true, a_pred, sample_weight=a_weight, level=scale+1)
    # TODO: just for a?
    _llap_f = llap_f(f_true, f_pred, sample_weight=f_weight, level=scale+1)
    _llap_b = llap_b(b_true, b_pred, sample_weight=b_weight, level=scale+1)

    return (
        _l1_a
        + _lc_a
        + _llap_a
        + 0.25 * (_l1_f + _l1_b + _lc_fb + _llap_f + _llap_b)
    ) * (1.5**scale / 1.5 ** (scales - 1))


def exp_mat_losses(scales=5):
    return [
        WeightedLossFunctionWrapper(
            partial(exp_mat_loss, scale=i)
        )
        for i in range(scales)
    ] + [None]
