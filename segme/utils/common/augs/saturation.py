import tensorflow as tf

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import convert
from segme.utils.common.augs.common import validate


def saturation(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or "saturation"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _saturation(x, factor),
            tf.identity,
            tf.identity,
        )


def _saturation(image, factor, name=None):
    with tf.name_scope(name or "saturation_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, "float32")

        image = tf.image.adjust_saturation(image, factor)

        return convert(image, dtype, saturate=True)
