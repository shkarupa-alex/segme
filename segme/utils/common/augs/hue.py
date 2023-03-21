import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def hue(image, masks, weight, prob, quality, name=None):
    with tf.name_scope(name or 'hue'):
        return apply(
            image, masks, weight, prob,
            lambda x: _hue(x, quality),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _hue(image, quality, name=None):
    with tf.name_scope(name or 'hue_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        image = tf.image.adjust_hue(image, quality)

        return convert(image, dtype, saturate=True)
