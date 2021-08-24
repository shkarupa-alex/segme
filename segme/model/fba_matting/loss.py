from keras.losses import MeanAbsoluteError
from ...loss import WeightedLossFunctionWrapper
from ...loss import ForegroundBackgroundExclusionLoss, GradientMeanSquaredError, LaplacianPyramidLoss


def l1_a(a_true, a_pred, sample_weight):
    return MeanAbsoluteError()(a_true, a_pred, sample_weight=sample_weight)


def l1_fb(f_true, b_true, f_pred, b_pred, sample_weight):
    loss = MeanAbsoluteError()(f_true, f_pred, sample_weight=sample_weight)
    loss += MeanAbsoluteError()(b_true, b_pred, sample_weight=sample_weight)
    loss /= 2.

    return loss


def lc_a(f_true, b_true, c_true, a_pred, sample_weight):
    _a_pred = a_pred / 255.
    c_pred = _a_pred * f_true + (1. - _a_pred) * b_true

    return MeanAbsoluteError()(c_true, c_pred, sample_weight=sample_weight)


def lc_fb(a_true, c_true, f_pred, b_pred, sample_weight):
    _a_true = a_true / 255.
    c_pred = _a_true * f_pred + (1. - _a_true) * b_pred

    return MeanAbsoluteError()(c_true, c_pred, sample_weight=sample_weight)


def lexcl_fb(f_pred, b_pred, sample_weight):
    return ForegroundBackgroundExclusionLoss()(f_pred, b_pred, sample_weight=sample_weight)


def lg_a(a_true, a_pred, sample_weight):
    return GradientMeanSquaredError()(a_true, a_pred, sample_weight=sample_weight)


def llap_a(a_true, a_pred, sample_weight):
    return LaplacianPyramidLoss()(a_true, a_pred, sample_weight=sample_weight)


def llap_fb(f_true, b_true, f_pred, b_pred, sample_weight):
    loss = LaplacianPyramidLoss()(f_true, f_pred, sample_weight=sample_weight)
    loss += LaplacianPyramidLoss()(b_true, b_pred, sample_weight=sample_weight)
    loss /= 2.

    return loss


def total_loss(afb_true, afb_pred, sample_weight=None):
    # FBA uses 2**(i) for llap scale factor https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270

    a_true, f_true, b_true = afb_true[..., 0:1], afb_true[..., 1:4], afb_true[..., 4:7]
    a_pred, f_pred, b_pred = afb_pred[..., 0:1], afb_pred[..., 1:4], afb_pred[..., 4:7]

    _a_true = a_true / 255.
    c_true = _a_true * f_true + (1. - _a_true) * b_true

    _l1_a = l1_a(a_true, a_pred, sample_weight)
    _lc_a = lc_a(f_true, b_true, c_true, a_pred, sample_weight)
    _lg_a = lg_a(a_true, a_pred, sample_weight)
    _llap_a = llap_a(a_true, a_pred, sample_weight)

    _l1_fb = l1_fb(f_true, b_true, f_pred, b_pred, sample_weight)
    _lc_fb = lc_fb(a_true, c_true, f_pred, b_pred, sample_weight)
    _lexcl_fb = lexcl_fb(f_pred, b_pred, sample_weight)
    _llap_fb = llap_fb(f_true, b_true, f_pred, b_pred, sample_weight)

    # Our final loss function is (2)
    # LFBα = L1α + Lcα + Lgα + Llapα + 0.25(L1FB + LcFB + LexclFB + LlapFB)

    return _l1_a + _lc_a + _lg_a + _llap_a + 0.25 * (_l1_fb + _lc_fb + _lexcl_fb + _llap_fb)


def fba_matting_loss():
    return WeightedLossFunctionWrapper(total_loss)
