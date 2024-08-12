import tensorflow as tf

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import convert
from segme.utils.common.augs.common import validate


def jpeg(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or "jpeg"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _jpeg(x, factor),
            tf.identity,
            tf.identity,
        )


def _jpeg(image, factor, name=None):
    with tf.name_scope(name or "jpeg_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, "uint8")

        image_ = tf.map_fn(
            lambda x: tf.image.adjust_jpeg_quality(x, factor), image
        )
        image_.set_shape(image.shape)

        return convert(image_, dtype, saturate=True)
