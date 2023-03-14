import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, validate


def equalize(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'equalize'):
        return apply(
            image, masks, weight, prob,
            lambda x: _equalize(x),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _equalize(image, name=None):
    with tf.name_scope(name or 'equalize_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'uint8', saturate=True)

        def _equalize_2d(image):
            histo = tf.histogram_fixed_width(image, [0, 255], nbins=256)

            step = histo[histo > 0]
            step = (tf.reduce_sum(step) - step[-1]) // 255

            lut = (tf.cumsum(histo) + (step // 2)) // tf.maximum(step, 1)
            lut = tf.concat([[0], lut[:-1]], 0)
            lut = tf.clip_by_value(lut, 0, 255)

            image_ = tf.gather(lut, image)

            apply = tf.cast(step > 0, image.dtype)
            image = image_ * apply + image * (1 - apply)

            return image

        batch, height, width, channel = tf.unstack(tf.shape(image))
        image_ = tf.cast(image, 'int32')
        image_ = tf.transpose(image_, [0, 3, 1, 2])
        image_ = tf.reshape(image_, [batch * channel, height, width])
        image_ = tf.map_fn(_equalize_2d, image_)
        image_ = tf.reshape(image_, [batch, channel, height, width])
        image_ = tf.transpose(image_, [0, 2, 3, 1])
        image_.set_shape(image.shape)
        image = tf.cast(image_, 'uint8')

        return convert(image, dtype, saturate=True)
