import tensorflow as tf


def autocontrast(images, name=None):
    with tf.name_scope(name or 'autocontrast'):
        images = tf.convert_to_tensor(images, name='images')
        if 4 != images.shape.rank:
            raise ValueError('Expecting `images` rank to be 4.')
        if 3 != images.shape[-1]:
            raise ValueError('Expecting `images` channels size to be 3.')

        dtype = images.dtype
        images = images if dtype in {tf.float16, tf.float32} else tf.image.convert_image_dtype(images, 'float32')

        lo = tf.reduce_min(images, axis=[1, 2], keepdims=True)
        hi = tf.reduce_max(images, axis=[1, 2], keepdims=True)

        images_ = tf.math.divide_no_nan(images - lo, hi - lo)

        apply = tf.cast(hi > lo, images.dtype)
        images = images_ * apply + images * (1 - apply)

        return tf.image.convert_image_dtype(images, dtype, saturate=True)
