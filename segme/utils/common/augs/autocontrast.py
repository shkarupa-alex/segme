import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def autocontrast(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'autocontrast'):
        return apply(
            image, masks, weight, prob,
            lambda x: _autocontrast(x),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _autocontrast(image, name=None):
    with tf.name_scope(name or 'autocontrast_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        lo = tf.reduce_min(image, axis=[1, 2], keepdims=True)
        hi = tf.reduce_max(image, axis=[1, 2], keepdims=True)

        image_ = tf.math.divide_no_nan(image - lo, hi - lo)

        mask = tf.cast(hi > lo, image.dtype)
        image = image_ * mask + image * (1 - mask)

        return convert(image, dtype, saturate=True)
