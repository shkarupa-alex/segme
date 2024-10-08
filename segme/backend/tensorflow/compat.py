import tensorflow as tf
from keras.src.backend import standardize_dtype
from keras.src.backend.tensorflow.core import convert_to_tensor


def l2_normalize(x, axis=-1, epsilon=1e-12):
    x = convert_to_tensor(x)
    return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)


def logdet(x):
    x = convert_to_tensor(x)
    return tf.linalg.logdet(x)


def saturate_cast(x, dtype):
    dtype = standardize_dtype(dtype)
    if isinstance(x, tf.SparseTensor):
        x_shape = x.shape
        x = tf.saturate_cast(x, dtype)
        x.set_shape(x_shape)
        return x
    else:
        return tf.saturate_cast(x, dtype)
