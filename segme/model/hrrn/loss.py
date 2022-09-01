import numpy as np
import tensorflow as tf
from keras.losses import MeanAbsoluteError, LossFunctionWrapper


def _loss_uncertainty(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)

    y_mean, y_var = tf.split(y_pred, 2, axis=-1)
    y_var = tf.clip_by_value(y_var, 0.01, 0.99)

    loss = tf.math.squared_difference(y_true, y_mean) / y_var  # softplus(y_var) ?
    # loss += tf.math.log(y_var) # in paper
    loss += tf.math.sqrt(2.0 * np.pi * y_var)  # in code
    loss *= 0.5

    # neg_log_likelihood = -y_pred.log_prob(y_true)

    return loss


def hrrn_losses():
    return [MeanAbsoluteError(), LossFunctionWrapper(_loss_uncertainty)]
