import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def gamma(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or 'gamma'):
        return apply(
            image, masks, weight, prob,
            lambda x: _gamma(x, factor),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _gamma(image, factor, name=None):
    with tf.name_scope(name or 'gamma_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        image = tf.image.adjust_gamma(image, factor)

        return convert(image, dtype, saturate=True)
