import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def contrast(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or 'contrast'):
        return apply(
            image, masks, weight, prob,
            lambda x: _contrast(x, factor),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _contrast(image, factor, name=None):
    with tf.name_scope(name or 'contrast_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        image = tf.image.adjust_contrast(image, factor)

        return convert(image, dtype, saturate=True)
