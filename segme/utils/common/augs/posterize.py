import tensorflow as tf

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import convert
from segme.utils.common.augs.common import validate


def posterize(image, masks, weight, prob, bits, name=None):
    with tf.name_scope(name or "posterize"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _posterize(x, bits),
            tf.identity,
            tf.identity,
        )


def _posterize(image, bits, name=None):
    with tf.name_scope(name or "posterize_"):
        image, _, _ = validate(image, None, None)
        if not tf.is_tensor(bits) and (bits > 8 or bits < 0):
            raise ValueError("Expecting `bits` to be in range [0, 8).")

        dtype = image.dtype
        image = convert(image, "uint8", saturate=True)

        image = tf.bitwise.right_shift(image, bits)
        image = tf.bitwise.left_shift(image, bits)

        return convert(image, dtype, saturate=True)
