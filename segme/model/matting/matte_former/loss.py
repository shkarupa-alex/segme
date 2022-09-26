import tensorflow as tf
from segme.loss import MeanSquaredRegressionError, LaplacianPyramidLoss, WeightedLossFunctionWrapper


def total_loss(afb_true, a_pred, sample_weight=None):
    a_true, f_true, b_true = tf.split(afb_true, [1, 3, 3], axis=-1)

    c_true = a_true * f_true + (1. - a_true) * b_true
    c_pred = a_pred * f_true + (1. - a_pred) * b_true

    l1 = MeanSquaredRegressionError()(a_true, a_pred, sample_weight=sample_weight)
    lap = LaplacianPyramidLoss(sigma=1.056, weight_pooling='max')(a_true, a_pred, sample_weight=sample_weight)
    comp = MeanSquaredRegressionError()(c_true, c_pred)

    return l1 * 2. + lap + comp


def matte_former_losses():
    return [None] + [WeightedLossFunctionWrapper(total_loss) for _ in range(3)]
