import tensorflow as tf
from segme.common.shape import get_shape
from segme.utils.common.augs.common import apply, convert, validate


def brightness(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or 'brightness'):
        return apply(
            image, masks, weight, prob,
            lambda x: _brightness(x, factor), tf.identity, tf.identity)


def _brightness(image, factor, name=None):
    with tf.name_scope(name or 'brightness_'):
        image, _, _ = validate(image, None, None)
        factor = tf.convert_to_tensor(factor)

        if factor.shape.rank:
            (batch,), _ = get_shape(image, axis=[0])
            factor = factor[:batch]

        dtype = image.dtype
        image = convert(image, 'float32')

        image = tf.image.adjust_brightness(image, factor)

        return convert(image, dtype, saturate=True)
