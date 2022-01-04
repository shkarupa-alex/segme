from keras.losses import MeanAbsoluteError
from ...loss import WeightedLossFunctionWrapper
from ...loss import ReflectionTransmissionExclusionLoss, GradientMeanSquaredError, LaplacianPyramidLoss


def l1_a(a_true, a_pred, sample_weight):
    return MeanAbsoluteError()(a_true, a_pred, sample_weight=sample_weight)


def l1_fb(f_true, b_true, f_pred, b_pred, sample_weight):
    loss = MeanAbsoluteError()(f_true, f_pred, sample_weight=sample_weight)
    loss += MeanAbsoluteError()(b_true, b_pred, sample_weight=sample_weight)

    return loss


def lc_a(f_true, b_true, c_true, a_pred, sample_weight):
    c_pred = a_pred * f_true + (1. - a_pred) * b_true

    return MeanAbsoluteError()(c_true, c_pred, sample_weight=sample_weight)


def lc_fb(a_true, c_true, f_pred, b_pred, sample_weight):
    c_pred = a_true * f_pred + (1. - a_true) * b_pred

    return MeanAbsoluteError()(c_true, c_pred, sample_weight=sample_weight)


def lexcl_fb(f_pred, b_pred, sample_weight):
    return ReflectionTransmissionExclusionLoss()(f_pred, b_pred, sample_weight=sample_weight)


def lg_a(a_true, a_pred, sample_weight):
    return GradientMeanSquaredError()(a_true, a_pred, sample_weight=sample_weight)


def llap_a(a_true, a_pred, sample_weight):
    return LaplacianPyramidLoss(sigma=1.075)(a_true, a_pred, sample_weight=sample_weight)


def llap_fb(f_true, b_true, f_pred, b_pred, sample_weight):
    loss = LaplacianPyramidLoss(sigma=1.075)(f_true, f_pred, sample_weight=sample_weight)
    loss += LaplacianPyramidLoss(sigma=1.075)(b_true, b_pred, sample_weight=sample_weight)

    return loss


def total_loss(afb_true, afb_pred, sample_weight=None, stage=0):
    a_true, f_true, b_true = afb_true[..., 0:1], afb_true[..., 1:4], afb_true[..., 4:7]
    a_pred, f_pred, b_pred = afb_pred[..., 0:1], afb_pred[..., 1:4], afb_pred[..., 4:7]

    _l1_a = l1_a(a_true, a_pred, sample_weight)
    _l1_fb = l1_fb(f_true, b_true, f_pred, b_pred, sample_weight)

    if 5 == stage:
        return _l1_a + 0.25 * _l1_fb

    c_true = a_true * f_true + (1. - a_true) * b_true
    _lc_a = lc_a(f_true, b_true, c_true, a_pred, sample_weight)
    _lc_fb = lc_fb(a_true, c_true, f_pred, b_pred, sample_weight)

    if 4 == stage:
        return _l1_a + _lc_a + 0.25 * (_l1_fb + _lc_fb)

    _llap_a = llap_a(a_true, a_pred, sample_weight)
    _llap_fb = llap_fb(f_true, b_true, f_pred, b_pred, sample_weight)

    if 3 == stage:
        return _l1_a + _lc_a + _llap_a + 0.25 * (_l1_fb + _lc_fb + _llap_fb)

    _lg_a = lg_a(a_true, a_pred, sample_weight)

    if 2 == stage:
        return _l1_a + _lc_a + _lg_a + _llap_a + 0.25 * (_l1_fb + _lc_fb + _llap_fb)

    _lexcl_fb = lexcl_fb(f_pred, b_pred, sample_weight)

    if 1 == stage:
        return _l1_a + _lc_a + _llap_a + 0.25 * (_l1_fb + _lc_fb + _lexcl_fb + _llap_fb)

    # 0 == stage
    # Our final loss function is (2)
    # LFBα = L1α + Lcα + Lgα + Llapα + 0.25(L1FB + LcFB + LexclFB + LlapFB)

    return _l1_a + _lc_a + _lg_a + _llap_a + 0.25 * (_l1_fb + _lc_fb + _lexcl_fb + _llap_fb)


def fba_matting_loss(stage=0):
    return WeightedLossFunctionWrapper(total_loss, stage=stage)
