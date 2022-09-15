import tensorflow as tf
from keras.losses import MeanAbsoluteError
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.laplacian_pyramid import LaplacianPyramidLoss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


def total_loss(afb_true, a_pred, sample_weight=None):
    a_true, f_true, b_true = tf.split(afb_true, [1, 3, 3], axis=-1)

    c_true = a_true * f_true + (1. - a_true) * b_true
    c_pred = a_pred * f_true + (1. - a_pred) * b_true

    l1 = MeanAbsoluteError(reduction=Reduction.NONE)(a_true, a_pred, sample_weight=sample_weight)
    l1 = tf.reduce_mean(l1, axis=[1, 2])

    lap = LaplacianPyramidLoss(sigma=1.056, reduction=Reduction.NONE)(a_true, a_pred, sample_weight=sample_weight)

    comp = MeanAbsoluteError(reduction=Reduction.NONE)(c_true, c_pred)
    comp = tf.reduce_mean(comp, axis=[1, 2])

    return l1 * 2. + lap + comp


def matte_former_losses():
    return [None] + [WeightedLossFunctionWrapper(total_loss) for _ in range(3)]
