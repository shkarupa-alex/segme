import cv2
import tensorflow as tf


def alpha_trimap(alpha, size, name=None):
    with tf.name_scope(name or 'alpha_trimap'):
        alpha = tf.convert_to_tensor(alpha, 'uint8')

        if 4 != alpha.shape.rank:
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 1 != alpha.shape[-1]:
            raise ValueError('Expecting `alpha` channels size to be 1.')

        if 'uint8' != alpha.dtype:
            raise ValueError('Expecting `alpha` dtype to be `uint8`.')

        if isinstance(size, tuple) and 2 == len(size):
            size = tf.random.uniform((), size[0], size[1] + 1, dtype='int32')

        if not isinstance(size, int) and not tf.is_tensor(size):
            raise ValueError('Expecting `size` to be a single margin or a tuple of [min; max] margins.')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))[..., None]
        kernel = tf.convert_to_tensor(kernel, 'int32')

        eroded = tf.cast(tf.equal(alpha, 255), 'int32')
        dilated = tf.cast(tf.greater(alpha, 0), 'int32')

        _, eroded, dilated = tf.while_loop(
            lambda i, *_: i < size,
            lambda i, e, d: (i + 1,
                             tf.nn.erosion2d(e, kernel, [1] * 4, 'SAME', 'NHWC', [1] * 4),
                             tf.nn.dilation2d(d, kernel, [1] * 4, 'SAME', 'NHWC', [1] * 4)),
            [0, eroded, dilated])

        trimap = tf.fill(tf.shape(alpha), 128)
        trimap = tf.where(tf.equal(eroded, 1 - size), 255, trimap)
        trimap = tf.where(tf.equal(dilated, size), 0, trimap)
        trimap = tf.cast(trimap, 'uint8')

        return trimap
