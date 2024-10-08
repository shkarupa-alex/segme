import numpy as np


def l2_normalize(x, axis=-1, epsilon=1e-12):
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_norm = np.sqrt(np.maximum(square_sum, epsilon))

    return x / x_norm


def logdet(x):
    raise NotImplementedError


def saturate_cast(x, dtype):
    raise NotImplementedError
