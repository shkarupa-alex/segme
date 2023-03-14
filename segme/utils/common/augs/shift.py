import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def shift(image, masks, weight, prob, factor=None, name=None):
    with tf.name_scope(name or 'shift'):
        return apply(
            image, masks, weight, prob,
            lambda x: _shift(x, factor),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _shift(image, factor=None, name=None):
    with tf.name_scope(name or 'shift_'):
        image, _, _ = validate(image, None, None)

        if factor is not None:
            factor = tf.convert_to_tensor(factor, image.dtype, name='factor')
            factor, _, _ = validate(factor, None, None)
            factor = convert(factor, 'float32')
        else:
            batch = tf.shape(image)[0]
            factor = tf.random.uniform([batch, 1, 1, image.shape[-1]], minval=-1 + 1e-5, maxval=1.)

        dtype = image.dtype
        image = convert(image, 'float32')

        image += factor

        return convert(image, dtype, saturate=True)
