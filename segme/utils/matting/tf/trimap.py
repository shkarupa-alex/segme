import cv2
import tensorflow as tf

from segme.common.shape import get_shape
from segme.utils.common.morph import dilate
from segme.utils.common.morph import erode


def alpha_trimap(alpha, size, name=None):
    with tf.name_scope(name or "alpha_trimap"):
        alpha = tf.convert_to_tensor(alpha, "uint8")

        if 4 != alpha.shape.rank:
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != alpha.dtype:
            raise ValueError("Expecting `alpha` dtype to be `uint8`.")

        if isinstance(size, tuple) and 2 == len(size):
            iterations = tf.random.uniform(
                [2], size[0], size[1] + 1, dtype="int32"
            )
            iterations = tf.unstack(iterations)
        elif isinstance(size, int):
            iterations = size, size
        else:
            raise ValueError(
                "Expecting `size` to be a single margin or a tuple of [min; max] margins."
            )

        eroded = tf.cast(tf.equal(alpha, 255), "int32")
        eroded = erode(eroded, 3, iterations[0])

        dilated = tf.cast(tf.greater(alpha, 0), "int32")
        dilated = dilate(dilated, 3, iterations[1])

        shape, _ = get_shape(alpha)
        trimap = tf.fill(shape, 128)
        trimap = tf.where(tf.equal(eroded, 1 - iterations[0]), 255, trimap)
        trimap = tf.where(tf.equal(dilated, iterations[1]), 0, trimap)
        trimap = tf.cast(trimap, "uint8")

        return trimap
