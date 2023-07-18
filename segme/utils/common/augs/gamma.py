import tensorflow as tf
from segme.common.shape import get_shape
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

        (batch,), _ = get_shape(image, axis=[0])
        invert = invert[:batch]

        image = tf.where(invert, 1. - image, image)
        image = tf.image.adjust_gamma(image, factor)
        image = tf.where(invert, 1. - image, image)

        return convert(image, dtype, saturate=True)
