import tensorflow as tf
from segme.utils.common.augs.common import apply, blend, convert, validate


def grayscale(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or 'grayscale'):
        return apply(
            image, masks, weight, prob,
            lambda x: _grayscale(x, factor),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _grayscale(image, factor, name=None):
    with tf.name_scope(name or 'grayscale_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        image_ = tf.image.rgb_to_grayscale(image)
        image_ = tf.image.grayscale_to_rgb(image_)
        image = blend(image, image_, factor)

        return convert(image, dtype, saturate=True)
