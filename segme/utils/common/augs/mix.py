import tensorflow as tf
from segme.common.shape import get_shape
from segme.utils.common.augs.common import apply, blend, convert, validate


def mix(image, masks, weight, prob, factor, color=None, name=None):
    with tf.name_scope(name or 'mix'):
        return apply(
            image, masks, weight, prob,
            lambda x: _mix(x, factor, color), tf.identity, tf.identity)


def _mix(image, factor, color=None, name=None):
    with tf.name_scope(name or 'mix_'):
        image, _, _ = validate(image, None, None)

        (batch,), _ = get_shape(image, axis=[0])
        factor = factor[:batch]

        if color is not None:
            color = tf.convert_to_tensor(color, image.dtype, name='color')
            color, _, _ = validate(color, None, None)
            color = color[:batch]
        else:
            color = tf.random.uniform([batch, 1, 1, image.shape[-1]])

        dtype = image.dtype
        image = convert(image, 'float32')
        color = convert(color, 'float32')

        image = blend(image, color, factor)

        return convert(image, dtype, saturate=True)
