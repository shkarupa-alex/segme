import tensorflow as tf

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import convert
from segme.utils.common.augs.common import validate


def gaussblur(image, masks, weight, prob, size, name=None):
    with tf.name_scope(name or "gaussblur"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _gaussblur(x, size),
            tf.identity,
            tf.identity,
        )


def _gaussblur(image, size, name=None):
    with tf.name_scope(name or "gaussblur_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, "float32")

        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        kernel = tf.cast(
            tf.range(-size // 2 + 1, size // 2 + 1), dtype="float32"
        )
        kernel = tf.exp(
            -tf.pow(kernel, 2) / (2.0 * tf.pow(tf.cast(sigma, "float32"), 2))
        )
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[:, None, None], [1, image.shape[-1], 1])

        image = tf.nn.depthwise_conv2d(
            image, kernel[None], [1] * 4, padding="SAME"
        )
        image = tf.nn.depthwise_conv2d(
            image, kernel[:, None], [1] * 4, padding="SAME"
        )

        return convert(image, dtype, saturate=True)
