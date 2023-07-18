import tensorflow as tf
from segme.common.shape import get_shape
from segme.utils.common.augs.common import apply, blend, convert, validate


def grayscale(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or 'grayscale'):
        return apply(
            image, masks, weight, prob,
            lambda x: _grayscale(x, factor), tf.identity, tf.identity)


def _grayscale(image, factor, name=None):
    with tf.name_scope(name or 'grayscale_'):
        image, _, _ = validate(image, None, None)
        factor = tf.convert_to_tensor(factor)

        if factor.shape.rank:
            (batch,), _ = get_shape(image, axis=[0])
            factor = factor[:batch]

        dtype = image.dtype
        image = convert(image, 'float32')

        image_ = tf.image.rgb_to_grayscale(image)
        image_ = tf.image.grayscale_to_rgb(image_)
        image = blend(image, image_, factor)

        return convert(image, dtype, saturate=True)
