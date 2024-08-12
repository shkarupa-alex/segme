import tensorflow as tf

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import convert
from segme.utils.common.augs.common import validate


def autocontrast(image, masks, weight, prob, name=None):
    with tf.name_scope(name or "autocontrast"):
        return apply(
            image, masks, weight, prob, _autocontrast, tf.identity, tf.identity
        )


def _autocontrast(image, name=None):
    with tf.name_scope(name or "autocontrast_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, "float32")

        lo = tf.reduce_min(image, axis=[1, 2], keepdims=True)
        hi = tf.reduce_max(image, axis=[1, 2], keepdims=True)

        image_ = tf.math.divide_no_nan(image - lo, hi - lo)
        image = tf.where(hi > lo, image_, image)

        return convert(image, dtype, saturate=True)
