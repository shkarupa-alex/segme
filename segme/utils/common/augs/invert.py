import tensorflow as tf
from segme.utils.common.augs.common import apply, validate


def invert(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'invert'):
        return apply(
            image, masks, weight, prob,
            lambda x: _invert(x),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _invert(image, name=None):
    with tf.name_scope(name or 'invert_'):
        image, _, _ = validate(image, None, None)

        max_val = image.dtype.max
        if image.dtype.is_floating:
            max_val = 1.

        image = max_val - image

        return image
