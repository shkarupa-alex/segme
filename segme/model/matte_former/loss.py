import tensorflow as tf
from keras.losses import MeanAbsoluteError
from segme.loss.laplacian_pyramid import LaplacianPyramidLoss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


def total_loss(afb_true, a_pred, sample_weight=None):
    a_true, f_true, b_true = tf.split(afb_true, [1, 3, 3], axis=-1)

    c_true = a_true * f_true + (1. - a_true) * b_true
    c_pred = a_pred * f_true + (1. - a_pred) * b_true

    l1 = MeanAbsoluteError()(a_true, a_pred, sample_weight=sample_weight)
    lap = LaplacianPyramidLoss(sigma=1.070482)(a_true, a_pred, sample_weight=sample_weight)
    comp = MeanAbsoluteError()(c_true, c_pred, sample_weight=sample_weight)

    return l1 * 2. + lap + comp


def matte_former_losses():
    return [None] + [WeightedLossFunctionWrapper(total_loss) for _ in range(3)]
