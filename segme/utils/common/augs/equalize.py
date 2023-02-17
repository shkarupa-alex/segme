import tensorflow as tf


def equalize(images, name=None):
    with tf.name_scope(name or 'equalize'):
        images = tf.convert_to_tensor(images, name='images')
        if 4 != images.shape.rank:
            raise ValueError('Expecting `images` rank to be 4.')
        if 3 != images.shape[-1]:
            raise ValueError('Expecting `images` channels size to be 3.')

        dtype = images.dtype
        if dtype.is_floating:
            # TODO: https://github.com/tensorflow/tensorflow/pull/54484
            images = tf.cast(tf.round(tf.clip_by_value(images, 0., 1.) * 255.), 'uint8')

        images = tf.image.convert_image_dtype(images, 'uint8', saturate=True)

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

        batch, height, width, channel = tf.unstack(tf.shape(images))
        images_ = tf.cast(images, 'int32')
        images_ = tf.transpose(images_, [0, 3, 1, 2])
        images_ = tf.reshape(images_, [batch * channel, height, width])
        images_ = tf.map_fn(_equalize_2d, images_)
        images_ = tf.reshape(images_, [batch, channel, height, width])
        images_ = tf.transpose(images_, [0, 2, 3, 1])
        images = tf.cast(images_, 'uint8')

        return tf.image.convert_image_dtype(images, dtype, saturate=True)
