import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def gamma(image, masks, weight, prob, factor, invert, name=None):
    with tf.name_scope(name or 'gamma'):
        return apply(
            image, masks, weight, prob,
            lambda x: _gamma(x, factor, invert), tf.identity, tf.identity)


def _gamma(image, factor, invert, name=None):
    with tf.name_scope(name or 'gamma_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        invert = tf.cast(invert, 'float32')

        image = (1. - image) * invert + image * (1. - invert)
        image = tf.image.adjust_gamma(image, factor)
        image = (1. - image) * invert + image * (1. - invert)

        return convert(image, dtype, saturate=True)
